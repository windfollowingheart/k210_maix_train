from modelscope.hub.api import HubApi
from modelscope.hub.api import ModelScopeConfig
from modelscope.msdatasets import MsDataset
api = HubApi()
#### token to user info
token = "34e88bae-3973-4574-9752-39bf1a02945e"
api.login(token)
# get user info
namespace, _ = ModelScopeConfig.get_user_info()
print(f"namespace: {namespace}")

namespace = 'windheart'
dataset_name = 'CelebA'

local_file_path = "/root/CelebA/Img/img_align_celeba.zip"

# 上传压缩后的文件
MsDataset.upload(
    object_name='Img/img_align_celeba.zip',
    local_file_path=local_file_path,
    dataset_name=dataset_name,
    namespace=namespace
)