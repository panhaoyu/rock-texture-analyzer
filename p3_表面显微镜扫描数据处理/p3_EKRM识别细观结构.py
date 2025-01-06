from rock_grain_identifier import RockGrainIdentifier
from rock_grain_identifier.group import RgiGroup


class Identifier(RockGrainIdentifier):
    groups = [
        RgiGroup(name='other', index=1, color='#00FF00'),
        RgiGroup(name='feldspar', index=2, color='#FF0000'),
        RgiGroup(name='quartz', index=3, color='#E1FF00'),
        RgiGroup(name='biotite', index=4, color='#000000'),
    ]

identifier = Identifier()