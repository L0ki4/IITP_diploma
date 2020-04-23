from skimage.transform import downscale_local_mean
from sklearn.preprocessing import LabelEncoder
from dpipe.dataset.csv import CSV
import numpy as np
import cv2


def binarize(labels):
    le = LabelEncoder()
    le.fit(sorted(list(set(labels))))
    return le.transform(labels)


class SCN(CSV):
    def __init__(self, path, filename, n_classes, slice_pad_num=32, diagnoses=(0, 1, 2), projections=(0, 2)):
        super().__init__(path=path, filename=filename, index_col='ID')

        if len(diagnoses) == 2:
            self.df = self.df[self.df.Diagnosis.isin(set(diagnoses))]
            self.df.Diagnosis = binarize(self.df.Diagnosis)

        self.df = self.df[self.df.projection.isin(set(projections))]

        self.slice_pad_num = slice_pad_num
        self.ids = tuple(self.df.index)
        self.models = tuple(self.df.ManufacturerModelName)
        self.projections = tuple(self.df.projection)

    def load_image(self, identifier):
        image = np.load(self.path + self.df.loc[identifier].PathToImage)
        image = cv2.resize(image, (256, 256))

        if image.shape[-1] > self.slice_pad_num:
            print(f'wrong shape {image.shape} identifier = {identifier}')
            raise Exception

        img_pad = np.pad(image, ((0, 0), (0, 0), (0, self.slice_pad_num - image.shape[-1])))
        return img_pad.transpose((2, 0, 1))

    def load_slice_amount(self, identifier):
        image = np.load(self.path + self.df.loc[identifier].PathToImage)
        return np.pad(np.ones(image.shape[-1]), (0, self.slice_pad_num-image.shape[-1])).astype(bool)

    def load_label(self, identifier):
        return self.get(identifier, 'Diagnosis')

    def load_model(self, identifier):
        return self.get(identifier, 'ManufacturerModelName')

    def load_projection(self, identifier):
        return self.get(identifier, 'projection')

    def load_labels(self, identifiers):
        return [self.load_label(i) for i in identifiers]

    def load_models(self, identifiers):
        return [self.load_model(i) for i in identifiers]

    def load_projections(self, identifiers):
        return [self.load_projection(i) for i in identifiers]


class SCN_aligned(CSV):
    def __init__(self, path, filename, downsample_factors, n_classes, cut_borders=(0, 22),
                 diagnoses=[0, 1, 2], projections=[0, 2], changed_models=None):

        super().__init__(path=path, filename=filename, index_col='ID')

        if len(diagnoses) == 2:
            self.df = self.df[self.df.Diagnosis.isin(set(diagnoses))]
            self.df.Diagnosis = binarize(self.df.Diagnosis)

        if changed_models is not None:
            self.df = self.df[self.df.ManufacturerModelName.isin(set(changed_models))]

        self.df = self.df[self.df.projection.isin(set(projections))]

        self.factors = downsample_factors
        self.ids = tuple(self.df.index)
        self.cut_borders = cut_borders
        self.models = tuple(self.df.ManufacturerModelName)
        self.projections = tuple(self.df.projection)

    def load_image(self, identifier):
        # print(self.df.loc[identifier].Diagnosis, ' ', self.df.loc[identifier].projection, ' ',
        #     self.df.loc[identifier].ManufacturerModelName)
        image = (np.array([np.load(self.path + self.df.loc[identifier].PathToImage)]))
        if self.df.loc[identifier].projection == 'sag':
            img = downsample(np.swapaxes(image, 1, 2)[:, :, self.cut_borders[0]:self.cut_borders[1], :], self.factors)
        else:  # 'tra'
            img = downsample(np.swapaxes(image, 2, 3)[:, :, self.cut_borders[0]:self.cut_borders[1], :], self.factors)
        # print(img.shape)
        return img

    def load_label(self, identifier):
        return self.get(identifier, 'Diagnosis')

    def load_model(self, identifier):
        return self.get(identifier, 'ManufacturerModelName')

    def load_projection(self, identifier):
        return self.get(identifier, 'projection')

    def load_labels(self, identifiers):
        return [self.load_label(i) for i in identifiers]

    def load_models(self, identifiers):
        return [self.load_model(i) for i in identifiers]

    def load_projections(self, identifiers):
        return [self.load_projection(i) for i in identifiers]


def downsample(image, factors):
    return downscale_local_mean(image, tuple(factors))
