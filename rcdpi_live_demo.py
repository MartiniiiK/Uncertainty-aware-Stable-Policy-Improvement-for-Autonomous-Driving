"""Separate live demonstration: actual RCDPI V1 or proposed USPI inference.
No official output paths are accepted; run only on an idle CARLA world.
"""
import argparse,fcntl,io,json,os,signal,sys,threading,time
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import carla,numpy as np
from PIL import Image
import evaluate_uspi as ev

class Hub:
    def __init__(self,policy):
        self.lock=threading.RLock();self.frames={};self.sensors=[];self.generation=0;self.ego_id=None;self.closed=False
        self.stats=dict(status='Starting',policy=policy,episode=0,step=0,speed=0.,cost=0.,**{'return':0.});self.policy=None
    def clear(self):
        with self.lock:self.generation+=1;self.frames.clear()
        for s in self.sensors:
            try:
                if s.is_alive:s.stop();s.destroy()
            except RuntimeError:pass
        self.sensors.clear()
    def callback(self,img,key,generation):
        rgb=np.frombuffer(img.raw_data,dtype=np.uint8).reshape(img.height,img.width,4)[:,:,:3][:,:,::-1]
        buf=io.BytesIO();Image.fromarray(np.ascontiguousarray(rgb)).save(buf,format='JPEG',quality=85)
        with self.lock:
            if generation==self.generation:self.frames[key]=buf.getvalue()
    def attach(self,env):
        self.clear()
        if env.world.get_settings().no_rendering_mode:raise RuntimeError('Rendering disabled')
        self.ego_id=env.ego.id
        with self.lock:self.stats.update(episode=self.stats['episode']+1,step=0,cost=0.,**{'return':0.})
        for key,loc,rot,width,height,fov in [('front',carla.Location(x=.7,y=-.3,z=1.6),carla.Rotation(pitch=-3),1280,720,95),('topdown',carla.Location(z=35),carla.Rotation(pitch=-90),960,640,90)]:
            bp=env.world.get_blueprint_library().find('sensor.camera.rgb')
            for k,v in [('image_size_x',width),('image_size_y',height),('fov',fov),('sensor_tick',.1)]:bp.set_attribute(k,str(v))
            cam=env.world.spawn_actor(bp,carla.Transform(loc,rot),attach_to=env.ego,attachment_type=carla.AttachmentType.Rigid)
            self.sensors.append(cam);g=self.generation;cam.listen(lambda img,k=key,g=g:self.callback(img,k,g));print('CAMERA READY:',key,flush=True)
    def step(self,env,result):
        _,reward,cost,done,info,nxt,actual=result
        with self.lock:
            self.stats.update(status='Running',step=self.stats['step']+1,speed=float(nxt[3])*3.6,cost=self.stats['cost']+float(cost),**{'return':self.stats['return']+float(reward)},selected_source=getattr(self.policy,'last',{}).get('selected_source','unknown'))
            if self.stats['step']>=50 and len(self.frames)!=2:raise RuntimeError('RGB cameras did not produce both frames')
            if self.stats['step']%50==0:print(self.stats,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['da_control','uspi'],default='da_control');p.add_argument('--checkpoint',required=True);p.add_argument('--control-checkpoint',required=True);p.add_argument('--uspi-config');p.add_argument('--episodes',type=int,default=10);p.add_argument('--seed',type=int,default=1000);p.add_argument('--device',default='cuda');p.add_argument('--confirm-idle',action='store_true');a=p.parse_args()
    if not a.confirm_idle:p.error('Stop all other CARLA controllers before --confirm-idle')
    hub=Hub('USPI (unvalidated demo)' if a.mode=='uspi' else 'RCDPI V1')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path in ('/stats','/dual_camera_status'):
                with hub.lock:data=json.dumps(hub.stats if path=='/stats' else {'ready':'topdown' in hub.frames}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            key={'/video_feed':'front','/video_feed_topdown':'topdown'}.get(path)
            if not key:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
            prev=None;last=0.
            try:
                while not hub.closed:
                    with hub.lock:frame=hub.frames.get(key)
                    if frame is not None and (frame is not prev or time.monotonic()-last>1):
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'+f'Content-Length: {len(frame)}\r\n\r\n'.encode()+frame+b'\r\n');self.wfile.flush();prev=frame;last=time.monotonic()
                    time.sleep(.04)
            except OSError:pass
    original_load,original_step=ev.load_policy,ev.checked_step
    def load(args):hub.policy=original_load(args);return hub.policy
    def step(env,sent,prev):
        t=time.monotonic()
        if env.ego.id!=hub.ego_id:hub.attach(env)
        result=original_step(env,sent,prev);hub.step(env,result);time.sleep(max(0.,.1-(time.monotonic()-t)));return result
    ev.load_policy=load;ev.checked_step=step
    out=Path('demo_runs')/('live_'+a.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    sys.argv=['evaluate_uspi.py','--mode',a.mode,'--checkpoint',a.checkpoint,'--control-checkpoint',a.control_checkpoint,'--out',str(out),'--episodes',str(a.episodes),'--seed',str(a.seed),'--steps','1000','--vehicles','20','--device',a.device]
    if a.uspi_config:sys.argv+=['--uspi-config',a.uspi_config]
    with open('/tmp/easycarla_evaluation.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        server=ThreadingHTTPServer(('127.0.0.1',8080),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        def stop(*_):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        try:ev.main()
        except KeyboardInterrupt:print('Demo stopped',flush=True)
        finally:hub.closed=True;hub.clear();server.shutdown();server.server_close()

if __name__=='__main__':main()
