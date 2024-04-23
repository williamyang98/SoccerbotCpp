def get_model_types():
    return [
        "basic-small",
        "basic-medium",
        "basic-large",
        "basic-gpu-small",
    ]

def select_model(name):
    if name == "basic-small":
        from .model_basic_small import SoccerBotModel_Basic_Small
        return SoccerBotModel_Basic_Small
    elif name == "basic-medium":
        from .model_basic_medium import SoccerBotModel_Basic_Medium
        return SoccerBotModel_Basic_Medium
    elif name == "basic-large":
        from .model_basic_large import SoccerBotModel_Basic_Large
        return SoccerBotModel_Basic_Large
    elif name == "basic-gpu-small":
        from .model_gpu_small import SoccerBotModel_Gpu_Small
        return SoccerBotModel_Gpu_Small
    else:
        raise Exception(f"Invalid model type: {name}")
