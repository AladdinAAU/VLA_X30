Split the navigation instruction into short ordered subgoals for a walking robot. Each subgoal is one thing to do or reach, and says how to recognise that it is finished.
Answer with JSON only: {"subgoals": [{"do": "...", "done_when": "..."}, ...]}. Use 1 to 6 subgoals. Keep the words of the instruction (landmarks, colours, directions, distances). Do not add steps that are not in the instruction.

Example instruction: "Walk down the corridor, turn left at the door and stop next to the table."
{"subgoals": [{"do": "walk down the corridor", "done_when": "the door is close ahead or beside the robot"}, {"do": "turn left at the door", "done_when": "the robot faces left of the door, the way beyond it is in view"}, {"do": "stop next to the table", "done_when": "the table is right beside the robot"}]}

When a step gives a distance or an angle, keep the number and make done_when the amount itself, e.g. {"do": "walk forward about 2 meters", "done_when": "the robot has moved about 2 meters forward"} or {"do": "turn left about 90 degrees", "done_when": "the robot has turned about 90 degrees left"}.
