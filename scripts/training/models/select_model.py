def get_model_types():
    return [
        "basic",
        "squeezenet",
        "basic-large",
        "squeezenet-large",
    ]

def select_model(name):
    if name == "basic":
        from .model_basic import SoccerBotModel_Basic
        return SoccerBotModel_Basic
    elif name == "squeezenet":
        from .model_squeezenet import SoccerBotModel_Squeezenet
        return SoccerBotModel_Squeezenet
    elif name == "basic-large":
        from .model_basic_large import SoccerBotModel_Basic_Large
        return SoccerBotModel_Basic_Large
    elif name == "squeezenet-large":
        from .model_squeezenet_large import SoccerBotModel_Squeezenet_Large
        return SoccerBotModel_Squeezenet_Large
    else:
        raise Exception(f"Invalid model type: {name}")
