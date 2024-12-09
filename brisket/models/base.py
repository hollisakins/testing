'''
Base model classes. 
'''

class BaseModel:
    '''Base model class. All brisket models should inherit from this class, or one of its subclasses.'''
    def __init__(self, params):
        self._build_defaults(params)
        self.params = params
    
    def _build_defaults(self, params):
        print('Warning: _build_defaults not implemented')
        pass
    
class BaseGriddedModel(BaseModel):
    '''Base class for gridded models, which must overwrite the ``_resample`` method.'''
    def _resample(self, wavelengths):
        raise NotImplementedError("Subclasses should implement this method")

class BaseFunctionalModel(BaseModel):
    '''Base class for functional models.'''
    def _resample(self, wavelengths):
        self.wavelengths = wavelengths

class BaseSourceModel(BaseModel):
    '''Base class for source models, which must overwrite the ``emit`` method.'''
    def __init__(self, params):
        self.model_type = 'source'
        super().__init__(params)
        

    def emit(self, params):
        raise NotImplementedError("Subclasses should implement this method")

class BaseAbsorberModel(BaseModel):
    '''Base class for absorber models, which must overwrite the ``absorb`` method.'''
    def __init__(self, params):
        self.model_type = 'absorber'
        super().__init__(params)
        

    def absorb(self, params):
        raise NotImplementedError("Subclasses should implement this method")

class BaseReprocessorModel(BaseModel):
    '''Base class for reprocessor models, which must overwrite the ``absorb`` and ``emit`` method.'''
    def __init__(self, params):
        self.model_type = 'reprocessor'
        super().__init__(params)

    def emit(self, params):
        raise NotImplementedError("Subclasses should implement this method")
    def absorb(self, params):
        raise NotImplementedError("Subclasses should implement this method")

