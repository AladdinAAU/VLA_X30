"""Spawn one person per outfit in front of the robot spawn pose to check the textures (sim container, GZ_PARTITION=e2e). python3 spawn_outfit_test.py [clean]"""
import sys
from gz.transport13 import Node
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
n = Node()
V = "file:///x30/configs/follow/models/Scrubs"
LIST = [("t_red", V + "Red", 3.2, 12.4), ("t_green", V + "Green", 3.2, 10.6), ("t_purple", V + "Purple", 4.6, 12.8), ("t_yellow", V + "Yellow", 4.6, 10.2),
        ("t_male", "file:///x30/datasets/external/fuel/models/MaleVisitorOnPhone", 6.0, 12.4), ("t_female", "file:///x30/datasets/external/fuel/models/FemaleVisitor", 6.0, 10.6)]
for nm, *_ in LIST:
    for suffix in ("", "_1", "_2"):
        e = Entity(); e.name = nm + suffix; e.type = Entity.MODEL
        n.request("/world/aws_hospital/remove", e, Entity, Boolean, 3000)
if "clean" not in sys.argv:
    for nm, uri, x, y in LIST:
        r = EntityFactory()
        r.sdf = f"<sdf version='1.9'><include><uri>{uri}</uri><name>{nm}</name><static>true</static><pose>{x} {y} 0 0 0 0</pose></include></sdf>"
        print(nm, n.request("/world/aws_hospital/create", r, EntityFactory, Boolean, 8000)[1].data)
