
# TODO 
VOCAB = {
    # 'base': 'lvis',
    'tracked': [
        # "tortilla",
        'tortilla pizza plain circular paper_plate quesadilla pancake: tortilla',
        # 'tortilla pizza plain circular paper_plate: tortilla',
        # "mug coffee tea: mug",
        "mug cup: mug",
        "bowl",
        "plate",
    ],
    'untracked': [
        "microwave_oven",
        # "tortilla plastic_bag packet ice_pack circular: tortilla_package",
        'tortilla_package',
        "paper_towel",
        "teabag",
        
        # "tortilla plastic_bag packet ice_pack circular: tortilla_package",
        'banana',
        "banana mushroom: banana_slice",
        'chopping_board clipboard place_mat tray: cutting_board',
        'knife',
        'spoon',
        'fork',
        'honey_jar',
        'peanut_butter_jar',
        'jelly_jar',
        'jar bottle can: jar',
        'jar_lid bottle_cap: jar_lid',
        'toothpicks',
        # 'floss',
        'watch', 'glove', 'person',

        'measuring_cup',
        'thermometer',
        'kettle',
        'kitchen_scale electronic_weight_scale: scale',
        'coffee_grinder',
        'paper_filter',
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
