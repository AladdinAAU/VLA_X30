# agent -> semantic_map

## 1. `/semantic_map/query` has no `id` filter (low priority)
`x30_agent` needs the record of one object by id (position and size of the object the model chose, for the finish guard and the
distance check). `SemanticMapperNode._svc_query` ignores unknown keys, so `{"id": 12}` silently returns the first 10 objects.
Workaround in `x30_agent/ros/ros_env.py`: query with `{"limit": 500}` and filter by id on the client side.
Request: accept `{"id": 12}` (or `"ids": [..]`) in `query`; no format change otherwise. Everything else the agent uses
(`summary.text`, `query.objects[].{id,label,position,size,bearing_only}`, `relations.text`, `find.text`) works as documented.
