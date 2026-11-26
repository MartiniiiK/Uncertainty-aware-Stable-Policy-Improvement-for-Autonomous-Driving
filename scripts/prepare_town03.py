"""Exclusive startup operation; never run during evaluation or live driving."""
import argparse, time, fcntl
import carla

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=2000);p.add_argument('--confirm-idle',action='store_true');a=p.parse_args()
if not a.confirm_idle:p.error('Stop ALL CARLA controllers before --confirm-idle')
with open('/tmp/easycarla_evaluation.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=carla.Client('127.0.0.1',a.port);c.set_timeout(15.)
    for attempt in range(20):
        try:w=c.get_world();break
        except RuntimeError as e:
            print('RPC waiting:',attempt+1,e,flush=True)
            if attempt==19:raise
            time.sleep(10)
    c.set_timeout(120.)
    if c.get_client_version()!=c.get_server_version():raise RuntimeError('CARLA client/server versions differ')
    s=w.get_settings();s.synchronous_mode=False;s.fixed_delta_seconds=None;s.no_rendering_mode=True;w.apply_settings(s);time.sleep(2)
    if w.get_map().name.split('/')[-1]!='Town03':w=c.load_world('Town03',reset_settings=False)
    for _ in range(10):print('Town03 tick:',w.wait_for_tick(30.).frame,flush=True)
    s=w.get_settings();s.synchronous_mode=False;s.fixed_delta_seconds=None;s.no_rendering_mode=False;w.apply_settings(s)
    for _ in range(10):print('Rendered tick:',w.wait_for_tick(30.).frame,flush=True)
    assert w.get_map().name.split('/')[-1]=='Town03' and not w.get_settings().no_rendering_mode
    print('TOWN03 READY',w.get_map().name,len(w.get_map().get_spawn_points()))
