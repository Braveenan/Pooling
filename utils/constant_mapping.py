from enum import Enum

class LabelKeywordMapping:
    LABEL2INDEX_SPEECHCOMMANDv1 = {
        '_silence_': 11,
        '_unknown_': 10,
        'down': 3,
        'go': 9,
        'left': 4,
        'no': 1,
        'off': 7,
        'on': 6,
        'right': 5,
        'stop': 8,
        'up': 2,
        'yes': 0
    }
    INDEX2LABEL_SPEECHCOMMANDv1 = {v: k for k, v in LABEL2INDEX_SPEECHCOMMANDv1.items()}
    
    LABEL2INDEX_VOXCELEB1 = {str(i): i-1 for i in range(1, 1252)}
    INDEX2LABEL_VOXCELEB1 = {v: k for k, v in LABEL2INDEX_VOXCELEB1.items()}
    
    LABEL2INDEX_IEMOCAP = {'ang': 2, 'hap': 1, 'neu': 0, 'sad': 3}
    INDEX2LABEL_IEMOCAP = {v: k for k, v in LABEL2INDEX_IEMOCAP.items()}

    # Group the mappings into a tuple
    speechcommand = (LABEL2INDEX_SPEECHCOMMANDv1, INDEX2LABEL_SPEECHCOMMANDv1)
    voxceleb = (LABEL2INDEX_VOXCELEB1, INDEX2LABEL_VOXCELEB1)
    iemocap = (LABEL2INDEX_IEMOCAP, INDEX2LABEL_IEMOCAP)

    @classmethod
    def get_label_mapping(cls, key):
        return getattr(cls, key)
    
class TaskKeywordMapping:
    ks = "Keyword Spotting"
    si = "Speaker Identification"
    er = "Emotion Recognition"
    
    @classmethod
    def get_task_name(cls, key):
        return getattr(cls, key, None)
    
class DatasetKeywordMapping:
    speechcommand = "SpeechCommand"
    voxceleb = "VoxCeleb"
    iemocap = "IEMOCAP"
    
    @classmethod
    def get_data_name(cls, key):
        return getattr(cls, key, None)
    