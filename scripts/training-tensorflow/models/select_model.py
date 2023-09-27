def get_model_types():
    return [
        "basic-small",
        "basic-medium",
    ]

def select_model(name):
    if name == "basic-small":
        from .model_basic_small import SoccerBotModel_Basic_Small
        return SoccerBotModel_Basic_Small
    elif name == "basic-medium":
        from .model_basic_medium import SoccerBotModel_Basic_Medium
        return SoccerBotModel_Basic_Medium
    else:
        raise Exception(f"Invalid model type: {name}")
