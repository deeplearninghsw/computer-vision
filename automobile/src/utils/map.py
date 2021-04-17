from collections import defaultdict


df_map_features = defaultdict(dict)

df_map_features["brand"] = [{
    "alfa-romero": 0,
    "audi": 1,
    "bmw": 2,
    "chevrolet": 3,
    "dodge": 4,
    "honda": 5,
    "isuzu": 6,
    "jaguar": 7,
    "mazda": 8,
    "mercedes-benz": 9,
    "mercury": 10,
    "mitsubishi": 11,
    "nissan": 12,
    "peugot": 13,
    "plymouth": 14,
    "porsche": 15,
    "renault": 16,
    "saab": 17,
    "subaru": 18,
    "toyota": 19,
    "volkswagen": 20,
    "volvo": 21
}]

df_map_features["fuel-type"] = [{
    "diesel": 0,
    "gas": 1
}]

df_map_features["fuel-type-back"] = [{
    0: "diesel",
    1: "gas"
}]

df_map_features["aspiration"] = [{
    "std": 0,
    "turbo": 1
}]

df_map_features["num-of-doors"] = [{
    "two": 2,
    "four": 4
}]

df_map_features["body-style"] = [{
    "hardtop": 0,
    "wagon": 1,
    "sedan": 2,
    "hatchback": 3,
    "convertible": 4
}]

df_map_features["drive-wheels"] = [{
    "4wd": 0,
    "fwd": 1,
    "rwd": 2
}]

df_map_features["engine-location"] = [{
    "front": 0,
    "rear": 1
}]

df_map_features["engine-type"] = [{
    "dohc": 0,
    "dohcv": 1,
    "l": 2,
    "ohc": 3,
    "ohcf": 4,
    "ohcv": 5,
    "rotor": 6
}]

df_map_features["num-of-cylinders"] = [{
    "eight": 8,
    "five": 5,
    "four": 4,
    "six": 6,
    "three": 3,
    "twelve": 12,
    "two": 2
}]

df_map_features["fuel-system"] = [{
    "1bbl": 0,
    "2bbl": 1,
    "4bbl": 2,
    "idi": 3,
    "mfi": 4,
    "mpfi": 5,
    "spdi": 6,
    "spfi": 7
}]
