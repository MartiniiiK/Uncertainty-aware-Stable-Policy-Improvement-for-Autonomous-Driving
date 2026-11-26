"""Read-only CARLA health check; never ticks a synchronous world."""
import carla
c=carla.Client('127.0.0.1',2000);c.set_timeout(15.)
print('Client/server:',c.get_client_version(),c.get_server_version());w=c.get_world();s=w.get_settings()
print('Map:',w.get_map().name,'spawn points:',len(w.get_map().get_spawn_points()))
print('Sync:',s.synchronous_mode,'Rendering:',not s.no_rendering_mode)
if not s.synchronous_mode:
    for _ in range(5):print('Tick:',w.wait_for_tick(10.).frame)
else:print('Read-only check: sync clock not driven')
