# modes -> semantic_map

Skills read the map through `/semantic_map/objects` (latched `SemanticObjectArray`, used as cache) and `/semantic_map/query` (`Json`):
request `{"label", "near": [x, y], "radius", "limit"}` and `{"id": n}` for one object; the parser accepts
`{"objects": [{"id", "label", "confidence", "position": [x, y, z] or {x,y,z}, "size": [sx, sy, sz], "bearing_only", "caption", "attributes", "room", "last_seen"}]}`
(or a bare list). `/semantic_map/summary` `{"pose": [x, y, yaw], "max_objects": n}` -> `{"text": ...}` is optional. If the query service is missing the topic is used.
Instances with `bearing_only: true` are never used as navigation targets (search may move toward them once to get a position).
