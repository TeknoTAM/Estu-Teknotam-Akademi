class Config:
    def __init__(self):
        self.model_type = "stable_version"
        # self.model_type = "standard_version"

        self.num_classes = 2  # background and people for hiphop data
        self.lr = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        self.input_size = (256, 256)

        self.images_path = "./DATA/train/images/"
        self.mask_paths = "./DATA/train/masks/"
