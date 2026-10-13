# Learner phase: teaching places by following a person (`x30_teach`)

Status 2026-10-12, simulator only (aws_hospital, kinematic locomotion). Typed/scripted sentences, no speech recognition yet. Numbers are from single runs (n = 1 per case): they show the loop works, they are not rates.

## 1. What it is
A new worker's first day: the robot follows a person (`follow_person`), the person says what places are ("This is Marcus.", "This is Alexander.", "Here we deliver stuff."), and the robot saves them in its semantic map as named rooms and spots. Afterwards the normal pipeline can use the names: "Go to Marcus." -> `locate_nav` reasoner reports the place -> planner drives.
```
person says a sentence ─▶ /teach/say {"text": ...}        (callable at any time; typed, scripted, later speech-to-text)
  1 where      the followed person's position (/follow/status x,y, not older than 3 s), else the robot (TF), else {"x","y"} in the request
  2 context    semantic mapper /semantic_map/place_context: the room-like region at that position, area/shape, objects nearby, taught places nearby
  3 LLM        Qwen3-VL-8B (reasoner role, text only, tool_choice required) picks ONE tool: save_place / rename_place / forget_place / ignore
  4 apply      /semantic_map/add_place | remove_place   (room = name the room region the person stands in, spot = a point)
  5 reply      short deterministic confirmation, returned and published on /teach/reply; every call appended to results/teach/teach_<date>.jsonl
```
Examples of what the model decides: "This is Marcus." -> save_place(name Marcus, kind room, owner Marcus); "Here we deliver stuff." -> save_place(name Delivery, kind spot, function 'deliver stuff'); "No, that's Marco." -> rename_place(Marcus -> Marco); "forget this place" -> forget_place; small talk -> ignore.

## 2. Pieces
* `x30_teach/interpreter.py` (ROS-free, 3 unit tests with a fake model): system prompt, the four tools, region phrase for the context ("inside a room (18 m2 ...)", "a corridor", "a large hall", "unmapped space"), parsing/cleaning of the model's tool call.
* `x30_teach/teach_node.py`: service `/teach/say` (x30_msgs/srv/Json), person position from `/follow/status`, calls to the mapper, reply, log. Launch: `ros2 launch x30_teach teach.launch.py use_sim_time:=true`. Model from `configs/models.yaml` (`reasoner_vlm`, http://vlm:8000/v1 inside the containers).
* `x30_semantic_map/places.py` (ROS-free, 5 tests): `Place` (name, kind room|spot, position, region id, area, owner, function, aliases, utterance), `add_place` (updates by name, case-insensitive; a hall or corridor region (> 80 m2 or named corridor) makes it a spot with a note), `room_region_at` (the region under the position if room-like; otherwise the nearest room-like region within 1.5 m **that is visible without crossing a wall**, so a person next to a wall of the room is in the room and a person in the corridor beside the wall is not), `apply_places` (re-applies names by position after every re-segmentation; names are `name_source user`, never overwritten by the rules), `context_at`, `summary_lines`. Places are saved/loaded with the map (`/semantic_map/save|load`) and a map reset clears them.
* Mapper services: `/semantic_map/add_place`, `/places`, `/remove_place`, `/place_context`. The map summary the reasoner reads has a block `Places (taught by people; use the name or its owner): - Marcus (room, owner Marcus, 7 m2) at (x,y), n m`.
* Scripts: `scripts/teach/say.py` (say something from the shell), `live_llm_check.py` (the model on 22 + 10 sentences), `run_tour.py` (the scripted tour), `run_use.sh` (navigation by taught names), `record_tour.sh` + `tour_demo.py` (rosbag demo).

## 3. Run it
Stack: `bash scripts/e2e/run_e2e.sh aws_hospital`; teach node (framework container); `bash scripts/follow/run_follow_stack.sh configs/follow/tour1.json` (follow node + the nurse). Say one sentence at a position: `python3 /x30/scripts/teach/say.py "This is Marcus." --world-xy 1.7 -3.25`. Tour: `python3 /x30/scripts/teach/run_tour.py --name tour_2` (lock command, the nurse walks into Room C, Room E and a corridor spot; at each stop, when the ground-truth person has stood there 3 s AND the follow node reports state `follow`, the sentence is sent with no position, so the module uses the followed person's position like for a real user). Use: `bash scripts/teach/run_use.sh tour_2`.

## 4. Measured
**Language model on its own** (`live_llm_check.py`, Qwen3-VL-8B, temperature 0, tool call forced; median 0.5 s per sentence): 22 sentences with context (rooms, corridor, hall; names, functions, corrections, deletions, small talk): 20/22 first, 21/22 after one prompt rule for bare names (the prompt was tuned on these sentences: that rule is a general one but the number is optimistic). The remaining miss ('This is the main hall.' in a 150 m2 hall: the model says room) is handled by the map code (> 80 m2 -> spot). 10 further sentences written before the model saw them: 10/10 (a small set).
**Tour in the simulator** (nurse = 'person', 120 s sim, robot follows 2.5 m behind, no other people): tour_1 (before the line-of-sight room rule): Marcus saved as the room in Room C (7 m2); Alexander saved as a *spot* (the person's position fell into the 392 m2 hall region and the model, told 'large hall', said spot); Delivery saved as a spot. tour_2 (after): Marcus = room in Room C (7.1 m2), Alexander = room in Room E (3.9 m2), Delivery = spot in the corridor with function 'deliver stuff'; all three: right ground-truth room, right kind, right owner/function, position source 'person', 0.7-0.9 s per sentence. The tool changes of tour_1 -> tour_2 were made after seeing tour_1; tour_2 is the same scenario, not a fresh one.
**Using the taught map** (`run_use.sh`, only the places and objects of tour_2, from the spawn, `locate_nav`): "Go to Marcus." -> inside Room C, 15.2 m, 58 s sim; "Go to Alexander's room." -> inside Room E, 22.5 m, 82 s; "Take this package to the delivery point." -> at the delivery spot, 25.9 m, 88 s. 3/3, two reasoner turns each (the model's first `report_target` call omits x and y, the harness answers with an error, the second call is right: a known quirk of the tool schema).

## 5. Limits and open points
* Text only: no speech-to-text, no wake word, no check that the speaker is the locked person; any text sent to `/teach/say` is believed.
* Room extent is the free-space region of the 2D map around the person (7 m2 and 3.9 m2 here for rooms of about 24 m2: furniture removes cells), not the walls of the room. Rooms the automatic segmentation cannot isolate (open doorways, large halls) become spots. Names stick to a region by position; if the region later splits or merges the name follows the position.
* One tour, three sentences, two rooms; no ambiguous sentences in the tour ('this is Marcus and that is Alexander', pronouns, 'the other room'); no names the model might know ('Doctor Weber' etc. were only in the language test); no re-teaching of a place from another position; no merge of two places with the same name in different rooms (the last one wins).
* The robot follows the person into a room only if the follow module manages the doorway; the tour worked here because the nurse stood where the robot could still see her (the sentence is only spoken when the follow node reports `follow`). Not tested: a person out of sight inside a room.
* The language model sees no image, only the text context; function and owner come from the sentence alone.
* Real robot: nothing run; the context needs real region segmentation and a real microphone/ASR path.

* **Robot position at the sentences (recorded tour):** at the first sentence the robot was 7.6 m behind the nurse in the hall outside Room C (the follow node still reported `follow`: she was tracked through the doorway). The place is anchored at the PERSON's tracked position, so naming worked, but this is not a robot standing inside the room; the same holds for the second room.
