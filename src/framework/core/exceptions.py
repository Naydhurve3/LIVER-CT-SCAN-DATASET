class MedSegXError(Exception):
    pass


class ConfigError(MedSegXError):
    pass


class DatasetError(MedSegXError):
    pass


class ModelError(MedSegXError):
    pass


class TrainingError(MedSegXError):
    pass


class EvaluationError(MedSegXError):
    pass


class RegistryError(MedSegXError):
    pass


class ReproducibilityError(MedSegXError):
    pass
