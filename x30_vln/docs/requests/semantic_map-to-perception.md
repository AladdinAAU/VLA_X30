# semantic_map -> perception

Observed in the sim chain run (docs/reports/semantic_map.md, section "Simulator chain"): in `lab_01` (flat-shaded SDF primitives, see
`docs/figures/semantic_map/sim_lab01_camera.jpg`) the YOLOE client delivered 699 detections in 247 s, of which 331 were `elevator`
(a plain wall or door), 85 `window`, 81 `mirror`, 32 `bed`; the lab has no elevator, mirror or bed. The mapper collapses them into one
instance per wall, but they stay in the map as wrong objects. Real classes of the world (`cabinet`, `server_rack`, `laptop`,
`fire_extinguisher`, `whiteboard`) were almost never reported.

Requests (none blocks the mapper):
1. A per-class minimum score or a "structure" threshold for `elevator`, `mirror`, `window`, `door`, `sign`, `curtain`, `pillar`, `bed`
   (large flat things score high on blank surfaces); the semantic map already requires 4 sightings but wall hallucinations are consistent.
2. Say in `docs/reports/perception.md` whether the sim camera frames are expected to work at all with the open-vocabulary model, or whether
   the simulated evaluation of the mapper must use `/gt/objects`-derived detections (a "oracle perception" publisher in the sim or eval area
   would let the mapper be scored without the detector).
3. Add `server_rack`, `filing cabinet`, `fire extinguisher`, `whiteboard`, `water cooler` to the default vocabulary if they are not
   there (the lab world contains them); the mapper's synonym groups already know most of them.
