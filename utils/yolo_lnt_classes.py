# Define the classes for your specific application
CUSTOM_CLASSES_LIST = [
    'job',
    'beam'
]

# For translating YOLO class ids (0 and 1) to SSD class ids (assuming they are 1 and 2 for consistency)
yolo_cls_to_ssd = [
    1, 2
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 2:
        return {i: n for i, n in enumerate(CUSTOM_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}