class MeteorologicalVariableType:
    def __init__(self, name, unit, factor_list, color_list):
        assert len(factor_list) == len(color_list)
        self.name = name
        self.unit = unit
        self.factor_list = factor_list
        self.color_list = color_list
