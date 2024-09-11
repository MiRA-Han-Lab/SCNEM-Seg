from cloudvolume import CloudVolume
from typing import List
import numpy as np
from scipy.io import loadmat
from .affine_transform import affine_trans_block
from scipy.ndimage import zoom
from connectomics.data.dataset import VolumeDataset
import os

class CVIO():
    def __init__(self, path, resolution, volume_size, target, chunk_size: List[int] = [512, 512, 128],  mode: str = 'read', pad_size: List[int] = [0, 0, 0],
                 data_scale: List[float] = [1.0, 1.0, 1.0],
                 affine: bool = False,
                 affine_matrix_path='../data/affine_matrix.mat',
                 **kwargs):

        self.kwargs = kwargs
        if mode == 'save':
            num_channels = 3 if target == 'DCV_CCV' or target == 'neuron' or target == 'defect' or target=='soma' else 1
            layer_type = 'segmentation' if target == 'type' else 'image'
            info = CloudVolume.create_new_info(
                num_channels=num_channels,
                layer_type=layer_type, # 'image' or 'segmentation'
                data_type='uint8', # can pick any popular uint
                encoding='raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
                resolution=resolution, # X,Y,Z values in nanometers
                voxel_offset=[0, 0, 0], # values X,Y,Z values in voxels
                chunk_size=chunk_size, # rechunk of image X,Y,Z in voxels
                volume_size=volume_size, # X,Y,Z size in voxels
            )
            if target == 'syn_ves':
                # If you're using amazon or the local file system, you can replace 'gs' with 's3' or 'file'
                vol = CloudVolume(path[0], info=info)
                vol.provenance.description = "Description of Data"
                vol.provenance.owners = ['email_address_for_uploader/imager'] # list of contact email addresses
    
                vol.commit_info() # generates gs://bucket/dataset/layer/info json file
                vol.commit_provenance() # generates gs://bucket/dataset/layer/provenance json file

                vol2 = CloudVolume(path[1], info=info)
                vol2.provenance.description = "Description of Data"
                vol2.provenance.owners = ['email_address_for_uploader/imager']  # list of contact email addresses

                vol2.commit_info()  # generates gs://bucket/dataset/layer/info json file
                vol2.commit_provenance()  # generates gs://bucket/dataset/layer/provenance json file
            else:
                vol = CloudVolume(path, info=info)
                vol.provenance.description = "Description of Data"
                vol.provenance.owners = ['email_address_for_uploader/imager']  # list of contact email addresses

                vol.commit_info()  # generates gs://bucket/dataset/layer/info json file
                vol.commit_provenance()  # generates gs://bucket/dataset/layer/provenance json file

        else:
            vol = CloudVolume(path, mip=resolution, parallel=1,
                                         progress=False)

            # ensuring that we always read something
            # (i.e. that we know what we're doing)
            vol.fill_missing = True
            vol.bounded = False
            # self.raw_version = 1


        self.cv = vol
        if mode == 'save' and target == 'syn_ves':
            self.cv2 = vol2
        self.target = target
        self.resolution = resolution
        self.pad_size = pad_size
        self.pad_size_constant = pad_size
        self.data_scale = data_scale
        self.affine = affine
        if self.affine:
            self.matrices = loadmat(affine_matrix_path)
            scale = 0.5
            self.affine_scale = int(affine_matrix_path.split('/')[-1].split('.')[0].split('downsample')[-1])*scale


    def writecv(self, coord: List[int], volume):
        #volume z y x
        z1, z2, y1, y2, x1, x2 = coord
        volume = np.transpose(volume, [3, 2, 1, 0])
        if self.target == 'mito':
            ########################
            mito = np.mean(volume[:, :, :, 0:-1], axis=-1)
            boundary = volume[:, :, :, -1]
            thre_bound = 128
            volume = mito*(boundary<thre_bound)
            ####################
            self.cv[x1:x2, y1:y2, z1:z2,0] = volume[:,:,:].astype('uint8')
        elif self.target == 'syn_ves':
            self.cv[x1:x2, y1:y2, z1:z2] = volume[:, :, :, 1:2]
            self.cv2[x1:x2, y1:y2, z1:z2] = volume[:,:,:,2:3]
        elif self.target == 'DCV_CCV':
            self.cv[x1:x2, y1:y2, z1:z2,:] = volume[:,:,:,:]
        elif self.target == 'neuron' or self.target == 'soma':
            self.cv[x1:x2, y1:y2, z1:z2, :] = volume
        elif self.target == 'defect':
            self.cv[x1:x2, y1:y2, z1:z2, :] = volume[:,:,:,1:4]
        elif self.target == 'type':
            self.cv[x1:x2, y1:y2, z1:z2, 0] = np.argmax(volume, axis=-1).astype('uint8')
        else:
            self.cv[x1:x2, y1:y2, z1:z2] = volume[:, :, :, 0:1]
            print('saving fininshed')

    def exists(self, coord: List[int]):
        z1, z2, y1, y2, x1, x2 = coord

        coord_exist = self.cv.exists(np.s_[x1:x2, y1:y2, z1:z2])
        for key in coord_exist:
            if not coord_exist[key]:
                return False
            if os.path.getsize(os.path.join(self.cv.cloudpath[7:],key+'.gz'))==0:
                return False
        if self.target == 'syn_ves':
            coord_exist = self.cv2.exists(np.s_[x1:x2, y1:y2, z1:z2])
            for key in coord_exist:
                if not coord_exist[key]:
                    return False
                if os.path.getsize(os.path.join(self.cv2.cloudpath[7:], key + '.gz')) == 0:
                    return False
        return True

    def load_return_chunk(self, coord):
        self.coord = coord
        if self.affine:

            h, w = 500, 500
            self.pad_size = [self.pad_size_constant[0], self.pad_size_constant[1]+h, self.pad_size_constant[2]+w]
        coord_p = coord + [-self.pad_size[0], self.pad_size[0],
                           -self.pad_size[1], self.pad_size[1],
                           -self.pad_size[2], self.pad_size[2]]
        print('load chunk: ', coord_p)

        z1, z2, y1, y2, x1, x2 = coord_p

        volume = [self.cv[x1:x2, y1:y2, z1:z2][:, :, :, 0].T]

        return volume

    def loadchunk(self, coord):
        r"""Load the chunk based on current coordinates and construct a VolumeDataset for processing.
        """
        print('load chunk: ', coord)
        self.coord = coord
        if self.affine:
            h, w = 500, 500
            self.pad_size = [self.pad_size_constant[0], self.pad_size_constant[1]+h, self.pad_size_constant[2]+w]

        coord_p = coord + [-self.pad_size[0], self.pad_size[0],
                                -self.pad_size[1], self.pad_size[1],
                                -self.pad_size[2], self.pad_size[2]]
        print('load padded chunk: ', coord_p)

        z1, z2, y1, y2, x1, x2 = coord_p

        volume = [self.cv[x1:x2, y1:y2, z1:z2][:,:,:,0].T]
        if not np.any(volume):
            return 0

        volume = self.maybe_scale(volume, order=1)  # linear for raw images
        ######### maybe apply affine transform ########
        volume = self.maybe_affine(volume, order=1)

        self.dataset = VolumeDataset(volume, None, None,
                                     mode='test',
                                     # specify chunk iteration number for training and -1 for inference
                                     iter_num=-1,
                                     **self.kwargs)
        return 1


    def maybe_scale(self, data, order=0):
        if (np.array(self.data_scale) != 1).any():
            print('scale:', self.data_scale)
            for i in range(len(data)):
                dt = data[i].dtype
                data[i] = zoom(data[i], self.data_scale,
                               order=order).astype(dt)

        return data



    def maybe_affine(self, data, order=1, invert=False):

        if not self.affine:
            return data
        z0, z1, y0, y1, x0, x1 = self.coord
        coord_p = self.coord + [-self.pad_size[0], self.pad_size[0],
                                -self.pad_size[1], self.pad_size[1],
                                -self.pad_size[2], self.pad_size[2]]
        z0o, z1o, y0o, y1o, x0o, x1o = coord_p  # region to crop

        scale = 4

        # for new block type affine matrix
        name = str(int(y0 // scale)) + '_' + str(int(y1 // scale)) + '_' + str(int(x0 // scale)) + '_' + str(
            int(x1 // scale)) + '_' + str(z0o) + '_' + str(z1o)
        matrix_cell = self.matrices[name]

        for i in range(len(data)):
            data[i] = affine_trans_block(data[i], np.copy(matrix_cell), affine_scale=self.affine_scale,
                                       Whether_invert=invert, order=order, n_threads=15)
        return data
