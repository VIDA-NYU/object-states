
# TODO 
VOCAB = {
    # 'base': 'lvis',
    'tracked': [
        # "tortilla",
        'tortilla pizza plain circular paper_plate quesadilla pancake: tortilla',
        # 'tortilla pizza plain circular paper_plate: tortilla',
        "mug coffee tea: mug",
        "bowl soup_bowl: bowl",
        "microwave_oven",
        "plate",

    ],
    'untracked': [
        "tortilla plastic_bag packet ice_pack circular: tortilla_package",
        'banana',
        "banana mushroom: banana_slice",
        'chopping_board clipboard place_mat tray: cutting_board',
        'knife',
        'jar bottle can: jar',
        'jar_lid bottle_cap: jar_lid',
        'toothpicks',
        # 'floss',
        'watch', 'glove', 'person',
    ],
    'equivalencies': {
        # equivalencies
        # 'can': 'bottle',
        'beer_can': 'bottle',
        'clipboard': 'chopping_board',
        'place_mat': 'chopping_board',
        'tray': 'chopping_board',
        
        # labels to ignore
        'table-tennis_table': 'IGNORE', 
        'table': 'IGNORE', 
        'dining_table': 'IGNORE', 
        'person': 'IGNORE',
        'watch': 'IGNORE',
        'glove': 'IGNORE',
        'magnet': 'IGNORE',
        'vent': 'IGNORE',
        'crumb': 'IGNORE',
        'nailfile': 'IGNORE',

        # not sure
        'handle': 'IGNORE',
    }

}
