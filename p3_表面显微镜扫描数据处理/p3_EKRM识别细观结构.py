from rock_grain_identifier import RockGrainIdentifier


class Identifier(RockGrainIdentifier):
    groups = [
        RockGrainIdentifier.Group(name='other', index=1, color='#00FF00'),
        RockGrainIdentifier.Group(name='feldspar', index=2, color='#FF0000'),
        RockGrainIdentifier.Group(name='quartz', index=3, color='#E1FF00'),
        RockGrainIdentifier.Group(name='biotite', index=4, color='#000000'),
    ]

identifier = Identifier()