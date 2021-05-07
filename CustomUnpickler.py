import pickle
from ModelArchitecture import *

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
    	model_classes = {
    		'SSD300': SSD300,
    		'VGGBase': VGGBase,
    		'AuxiliaryConvolutions': AuxiliaryConvolutions,
    		'PredictionConvolutions': PredictionConvolutions
    	}
    	return model_classes[name] if name in model_classes else super().find_class(module, name);
