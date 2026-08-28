class MTCNN:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "mtcnn detector requires tensorflow. "
            "Install tensorflow or use retinaface detector instead."
        )
