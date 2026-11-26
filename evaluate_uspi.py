"""Town03 pilot diagnostics; separate from existing evaluator and checkpoints.
Run only on a dedicated CARLA world, with no other controller.
"""
import argparse, csv, inspect, json, math, random, time, fcntl
from pathlib import Path
import numpy as np
import torch
from da_dql import build_dql, flatten
from da_dql_v2_policy import Policy

MODES = ('rule','official','dql','actor','full','da_control','uspi')
CONTROL_MEANS = {
    'candidate_count': 'mean_candidate_count',
    'accepted': 'mean_accepted_candidates',
    'supported_count': 'mean_supported_candidates',
    'selected_support_count': 'mean_selected_support_count',
    'selected_rule': 'rule_selection_fraction',
    'reward_improvement': 'mean_predicted_reward_improvement',
    'progress_improvement': 'mean_predicted_progress_improvement',
    'cost_improvement': 'mean_predicted_cost_improvement',
    'best_cost_improvement': 'mean_best_predicted_cost_improvement',
    'jerk_improvement': 'mean_predicted_jerk_improvement',
}


def rule_policy(obs):
    # Identical rule controller to the uploaded web_demo.py.
    speed = float(obs['ego_state'][3])
    waypoints = obs['waypoints'].reshape(-1,3)
    target = waypoints[min(3,len(waypoints)-1)]
    heading = math.atan2(float(target[1]),max(float(target[0]),0.1))
    steer = np.clip(1.5*heading,-0.8,0.8)
    throttle = np.clip(0.25*(8.0-speed),0,0.8)
    brake = np.clip(0.35*(speed-8.0),0,1)
    front = float(obs['ego_state'][7])
    if 0 < front < 7:
        throttle=0.0; brake=max(brake,0.6)
    return np.array([throttle,steer,brake],dtype=np.float32)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_policy(args):
    if args.mode=='rule': return rule_policy
    if args.mode=='uspi':
        from uspi_policy import USPIPolicy, StableConfig
        cfg=StableConfig(**json.loads(Path(args.uspi_config).read_text())) if args.uspi_config else StableConfig()
        return USPIPolicy(args.checkpoint,args.control_checkpoint,args.device,cfg)
    if args.mode in ('da_control','uspi'):
        from da_control_policy import ControlPolicy
        return ControlPolicy(args.checkpoint, args.control_checkpoint, args.device,
                             epsilon_cost=args.epsilon_cost)
    if args.mode=='official':
        m=build_dql(torch.device(args.device))
        folder=Path(args.official_dir)
        for name in ('actor','critic'):
            path=folder/f'{name}_{args.model_id}.pth'
            getattr(m,name).load_state_dict(torch.load(path,map_location=args.device,weights_only=True))
            getattr(m,name).eval()
        # Official loader does not restore the target critic; use saved online critic.
        m.critic_target.load_state_dict(m.critic.state_dict()); m.critic_target.eval()
        return lambda obs: m.sample_action(flatten(obs))
    root=Path(args.run)
    ckpt=Path(args.checkpoint) if args.checkpoint else root/('aware.pth' if args.mode in ('actor','full') else 'dql.pth')
    return Policy(str(ckpt),mode=args.mode,device=args.device)


def bounded(a):
    a=np.asarray(a,dtype=np.float32).reshape(3)
    if not np.isfinite(a).all(): raise FloatingPointError('Nonfinite action')
    return np.clip(a,[0,-1,0],[1,1,1]).astype(np.float32)


def net_action(a):
    # Execution-only ablation. Not a physical acceleration model or safety shield.
    u=float(a[0]-a[2]); return np.array([max(u,0),a[1],max(-u,0)],dtype=np.float32)


def overlap(a): return int(a[0]>.1 and a[2]>.1)


class IntegrityError(RuntimeError):
    pass


def check_executable_action(action):
    if (not np.isfinite(action).all() or np.any(action < [0., -1., 0.])
            or np.any(action > 1.) or (action[0] > 0. and action[2] > 0.)):
        raise IntegrityError(f'Invalid DA-DQL command: {action.tolist()}')


def checked_snapshot(env):
    snap = env.world.get_snapshot()
    if not env.ego.is_alive or snap.find(env.ego.id) is None:
        raise IntegrityError(f'Ego actor {env.ego.id} missing at frame {snap.frame}')
    if not env.world.get_settings().synchronous_mode:
        raise IntegrityError('World unexpectedly switched out of synchronous mode')
    return snap


def checked_step(env, sent, prev):
    before = checked_snapshot(env)
    obs, reward, cost, done, info = env.step(sent)
    after = checked_snapshot(env)
    if after.frame != before.frame + 1:
        raise IntegrityError(f'Unexpected frame advance: {before.frame} -> {after.frame}; check other controllers')
    nxt = np.asarray(obs['ego_state'])
    ctrl = env.ego.get_control()
    actual = np.array([ctrl.throttle, ctrl.steer, ctrl.brake], dtype=np.float32)
    if not np.isfinite(nxt).all() or not np.isfinite(actual).all() or not np.isfinite([reward, cost]).all():
        raise IntegrityError('Nonfinite observation, control, reward or cost')
    if np.any(actual < np.array([0.,-1.,0.])-1e-5) or np.any(actual > 1.+1e-5):
        raise IntegrityError(f'Invalid CARLA control readback: {actual.tolist()}')
    dist = float(np.linalg.norm(nxt[:2]-prev[:2]))
    if dist > max(5., .1*max(float(prev[3]),float(nxt[3]))+2.):
        raise IntegrityError(f'Unexpected ego displacement: {dist:.3f} m in one step')
    return obs, reward, cost, done, info, nxt, actual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True); p.add_argument('--run',default='da_runs/pilot42')
    p.add_argument('--checkpoint'); p.add_argument('--out',required=True); p.add_argument('--town',default='Town03')
    p.add_argument('--uspi-config', help='JSON StableConfig overrides')
    p.add_argument('--control-checkpoint', help='Rule-policy evaluation critics for da_control')
    p.add_argument('--epsilon-cost', type=float, default=0.,
                   help='Allowed loss of predicted cost reduction in discounted cost units')
    p.add_argument('--episodes',type=int,default=10); p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--vehicles',type=int,default=20); p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--port',type=int,default=2000); p.add_argument('--device',default='cuda')
    p.add_argument('--official-dir',default='example/params_dql'); p.add_argument('--model-id',type=int,default=200)
    p.add_argument('--control',choices=('raw','net'),default='raw')
    args=p.parse_args()
    if min(args.episodes,args.steps)<1 or args.vehicles<0: p.error('Invalid evaluation configuration')
    if args.mode in ('da_control','uspi') and (not args.checkpoint or not args.control_checkpoint):
        p.error('da_control requires --checkpoint for the DQL actor and --control-checkpoint for the rule critics')
    if not math.isfinite(args.epsilon_cost) or args.epsilon_cost < 0.:
        p.error('--epsilon-cost must be finite and nonnegative')
    if args.control=='net' and args.mode not in ('dql','full'): p.error('net is a separate diagnostic for dql/full only')
    import carla, gym, easycarla
    from easycarla.envs.carla_env import CarlaEnv
    if 'Reusing existing map' not in inspect.getsource(CarlaEnv.__init__):
        raise RuntimeError('Map-reuse patch missing; refusing to reload world')
    c=carla.Client('127.0.0.1',args.port); c.set_timeout(60.)
    world=c.get_world(); town=world.get_map().name.split('/')[-1]
    if town!=args.town: raise RuntimeError(f'Expected {args.town}, got {town}; no world reloaded')
    seed_all(args.seed); policy=load_policy(args)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'Use empty output directory: {out}')
    params={'number_of_vehicles':args.vehicles,'number_of_walkers':0,'dt':.1,'ego_vehicle_filter':'vehicle.tesla.model3','surrounding_vehicle_spawned_randomly':True,'port':args.port,'town':town,'max_time_episode':args.steps,'max_waypoints':12,'visualize_waypoints':False,'desired_speed':8,'max_ego_spawn_times':200,'view_mode':'follow','traffic':'off','lidar_max_range':50.,'max_nearby_vehicles':5}
    (out/'config.json').write_text(json.dumps({**vars(args),'params':params,'net_note':'post-selection diagnostic only; gate not re-evaluated','tm_note':'seeded, existing env reset toggles async/sync; not exact replay'},indent=2))
    env=gym.make('carla-v0',params=params,disable_env_checker=True).unwrapped
    old=env._get_obs
    def get_obs():
        deadline=time.monotonic()+15.
        while env.lidar_data is None:
            if time.monotonic()>deadline: raise TimeoutError('No initial LiDAR frame')
            time.sleep(.01)
        return old()
    env._get_obs=get_obs
    tm=c.get_trafficmanager(8000)
    original_spawns=list(env.vehicle_spawn_points)
    fields=['episode','seed','steps','return','cost','collision','offroad','mean_speed_kmh','distance_m','near_stop_fraction','raw_overlap_fraction','executed_overlap_fraction','mean_abs_delta_steer','mean_abs_delta_longitudinal','mean_inference_ms','mean_accepted_fraction','fallback_fraction','no_feasible_fraction','selected_pass_fraction','mean_predicted_cost','mean_validity','spawned_vehicles','initial_x','initial_y','initial_yaw']
    trace=['episode','seed','step','x','y','speed_mps','front_distance','raw_throttle','raw_steer','raw_brake','bounded_throttle','bounded_steer','bounded_brake','executed_throttle','executed_steer','executed_brake','reward','cost','collision','offroad','next_speed_mps','q','accepted','candidate_count','fallback','no_feasible','selected_passes','predicted_cost','validity','collision_probability','offroad_probability']
    if args.mode=='uspi':
        from dataclasses import asdict
        (out/'uspi_config.json').write_text(json.dumps(asdict(policy.config),indent=2)+'\n')
    integrity_failed=False
    fields.extend(('mean_planar_jerk_mps3', 'route_progress_m'))
    trace.extend(('planar_jerk_mps3', 'route_progress_m'))
    if args.mode in ('da_control','uspi'):
        fields.extend(CONTROL_MEANS.values())
        trace.extend(k for k in CONTROL_MEANS if k not in trace)
        trace.append('selected_source')
        if args.mode=='uspi':
            trace.extend(('gate_support_count','gate_advantage_count','gate_continuity_count','gate_trajectory_count','selected_lateral_proxy_m','selected_heading_proxy_rad','fallback_reason'))
    try:
        with (out/'episodes.csv').open('w',newline='') as fp, (out/'actions.csv').open('w',newline='') as tf:
            w=csv.DictWriter(fp,fieldnames=fields); w.writeheader(); fp.flush()
            tw=csv.DictWriter(tf,fieldnames=trace); tw.writeheader(); tf.flush()
            for episode in range(1,args.episodes+1):
                seed=args.seed+episode-1; seed_all(seed);
                if hasattr(policy,'reset'): policy.reset()
                tm.set_random_device_seed(seed)
                env.vehicle_spawn_points=list(original_spawns); env.lidar_data=None
                obs=env.reset(); initial=np.asarray(obs['ego_state']).copy()
                re=cost=distance=0.; collision=offroad=False; speeds=[]; times=[]; accepts=[]; fallbacks=[]; no_feasible=[]; selected_pass=[]; costs=[]; valid=[]; raw_over=[]; exec_over=[]; applied=[]
                jerk_sum = route_progress = 0.
                control_sums = dict.fromkeys(CONTROL_MEANS, 0.)
                control_counts = dict.fromkeys(CONTROL_MEANS, 0)
                for step in range(1,args.steps+1):
                    prev=np.asarray(obs['ego_state']).copy()
                    waypoints = np.asarray(obs['waypoints']).reshape(-1, 3)
                    route_heading = float(prev[2] + waypoints[0, 2]) if np.any(waypoints) else None
                    t=time.perf_counter(); raw=np.asarray(policy(obs),dtype=np.float32).reshape(3); times.append((time.perf_counter()-t)*1000)
                    if args.mode in ('da_control','uspi'):
                        check_executable_action(raw)
                        a = sent = raw
                    else:
                        a=bounded(raw); sent=net_action(a) if args.control=='net' else a
                    v=getattr(policy,'last',{}).copy()
                    obs,r,co,d,info,nxt,actual=checked_step(env,sent,prev)
                    jerk = float(np.linalg.norm(nxt[5:7]-prev[5:7])/params['dt'])
                    jerk_sum += jerk
                    progress = 0.
                    if route_heading is not None:
                        dx, dy = nxt[:2] - prev[:2]
                        progress = float(dx*math.cos(route_heading) + dy*math.sin(route_heading))
                    route_progress += progress
                    if args.mode in ('da_control','uspi'):
                        policy.observe_executed(actual)
                        for key in CONTROL_MEANS:
                            value = v.get(key)
                            if value is not None and math.isfinite(value):
                                control_sums[key] += value
                                control_counts[key] += 1
                    re+=float(r); cost+=float(co); collision|=bool(info.get('is_collision',False)); offroad|=bool(info.get('is_off_road',False)); speeds.append(float(nxt[3])*3.6)
                    distance+=float(np.linalg.norm(nxt[:2]-prev[:2])); raw_over.append(overlap(a)); exec_over.append(overlap(actual)); applied.append(actual)
                    if 'candidate_count' in v: accepts.append(v['accepted']/v['candidate_count'])
                    if 'fallback' in v: fallbacks.append(int(v['fallback']))
                    if 'predicted_cost' in v and math.isfinite(v['predicted_cost']): costs.append(v['predicted_cost'])
                    if 'validity' in v: valid.append(v['validity'])
                    if 'no_feasible' in v: no_feasible.append(int(v['no_feasible']))
                    if 'selected_passes' in v: selected_pass.append(int(v['selected_passes']))
                    row=dict(episode=episode,seed=seed,step=step,x=float(prev[0]),y=float(prev[1]),speed_mps=float(prev[3]),front_distance=float(prev[7]),reward=float(r),cost=float(co),collision=int(info.get('is_collision',False)),offroad=int(info.get('is_off_road',False)),next_speed_mps=float(nxt[3]))
                    for prefix,action in [('raw',raw),('bounded',a),('executed',actual)]:
                        for key,value in zip(('throttle','steer','brake'),action): row[f'{prefix}_{key}']=float(value)
                    row['planar_jerk_mps3'] = jerk
                    row['route_progress_m'] = progress
                    row.update({k:v[k] for k in trace if k in v}); tw.writerow(row)
                    if step%100==0: tf.flush()
                    if d: break
                applied=np.asarray(applied); dif=np.diff(applied,axis=0)
                row={'episode':episode,'seed':seed,'steps':step,'return':re,'cost':cost,'collision':int(collision),'offroad':int(offroad),'mean_speed_kmh':np.mean(speeds),'distance_m':distance,'near_stop_fraction':np.mean(np.asarray(speeds)<3.6),'raw_overlap_fraction':np.mean(raw_over),'executed_overlap_fraction':np.mean(exec_over),'mean_abs_delta_steer':np.abs(dif[:,1]).mean() if len(dif) else '', 'mean_abs_delta_longitudinal':np.abs(dif[:,0]-dif[:,2]).mean() if len(dif) else '', 'mean_inference_ms':np.mean(times),'mean_accepted_fraction':np.mean(accepts) if accepts else '', 'fallback_fraction':np.mean(fallbacks) if fallbacks else '', 'no_feasible_fraction':np.mean(no_feasible) if no_feasible else '', 'selected_pass_fraction':np.mean(selected_pass) if selected_pass else '', 'mean_predicted_cost':np.mean(costs) if costs else '', 'mean_validity':np.mean(valid) if valid else '', 'spawned_vehicles':len(env.spawned_vehicles),'initial_x':float(initial[0]),'initial_y':float(initial[1]),'initial_yaw':float(initial[2])}
                row['mean_planar_jerk_mps3'] = jerk_sum / step
                row['route_progress_m'] = route_progress
                if args.mode in ('da_control','uspi'):
                    row.update({label: control_sums[key]/control_counts[key] if control_counts[key] else ''
                                for key, label in CONTROL_MEANS.items()})
                w.writerow(row); fp.flush(); tf.flush(); print(row,flush=True)
    except IntegrityError as exc:
        integrity_failed=True
        (out/'INVALID.txt').write_text(str(exc)+'\n')
        print('INTEGRITY FAILURE:',exc,flush=True)
        raise
    finally:
        if not integrity_failed:
            try:
                env._set_synchronous_mode(False)
                env._clear_all_actors(['sensor.camera.rgb','sensor.lidar.ray_cast','sensor.other.collision','vehicle.*'])
            finally: env.close()
        else:
            print('Skipped global actor cleanup after integrity failure; inspect other CARLA clients before restarting.',flush=True)
    print('Evaluation complete:',out/'episodes.csv',flush=True)
if __name__=='__main__':
    # Advisory lock: all evaluators using this entrypoint share one CARLA lock.
    # Other (older) scripts and demos must still be stopped manually.
    with open('/tmp/easycarla_evaluation.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another protected CARLA evaluation is running')
        main()

