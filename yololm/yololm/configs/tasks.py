spi_datasets = [
    ###### ROC ######
    # {
    #     'type': 'coco_det',
    #     'vis_root': './data/coco',
    # },
    {
        'type': 'objects365',
        'cache_path': 'data/Objects365v1/train.cache.npy',
    },
    {
        'type': 'flickr',
        'cache_path': 'data/flickr/train.cache.npy',
    },
    {
        'type': 'mixed_grounding',
        'cache_path': 'data/mixed_grounding/train.cache.npy',
    },
    # {
    #     "type": "PartDataset",
    #     "ann_file": "./data/part_data/pascal_part/pascalpart_train.json",
    #     "img_prefix": "./data/part_data/pascal_part/VOCdevkit/VOC2010/JPEGImages"
    # },
    # {
    #     "type": "PartDataset",
    #     "ann_file": "./data/part_data/partImagenet/partImagenet_train_format.json",
    #     "img_prefix": "./data/part_data/partImagenet/train"
    # },
    # {
    #     'type': 'flickr30k',
    #     'ann_file': './data/mdetr_annotations/final_flickr_mergedGT_train.json',
    #     'img_prefix': './data/flickr30k-images/',
    #     'task_str': '<roc>',
    # },
    # {
    #     'type': 'mixed_grounding',
    #     'ann_file': './data/mixed_grounding/annotations/final_mixed_train_no_coco.json',
    #     'img_prefix': './data/mixed_grounding/gqa/images',
    #     'task_str': '<roc>',
    # },
    ###### CAP ######
    # {
    #     "type": "VGDATA",
    #     'ann_file': './data/visual_genome/train.json',
    #     'img_prefix': './data/visual_genome/vg_all',
    # },
    # {
    #     'type': 'flickr30k',
    #     'ann_file': './data/mdetr_annotations/final_flickr_mergedGT_train.json',
    #     'img_prefix': './data/flickr30k-images/',
    #     'task_str': '<cap>',
    # },
    # {
    #     'type': 'mixed_grounding',
    #     'ann_file': './data/mixed_grounding/annotations/final_mixed_train_no_coco.json',
    #     'img_prefix': './data/mixed_grounding/gqa/images',
    #     'task_str': '<cap>',
    # },
    # {
    #     "type": "RefCOCOG",
    #     'ann_file': './data/mdetr_annotations/finetune_refcocog_train.json',
    #     'img_prefix': './data/coco_all/',
    # },
    ###### VQA ######
    # {
    #     "type": "single_vcr",
    #     "ann_file": "./data/vcr/train.jsonl",
    #     "img_prefix": "./data/vcr/vcr1images"
    # },
    # {
    #     "type": "OspreyPartLevel",
    #     "ann_file": "./data/osprey-724k/osprey_part_level.json",
    #     "img_prefix": "./data/coco_all/"
    # },
    # {
    #     "type": "OspreyShortForm",
    #     "ann_file": "./data/osprey-724k/osprey_short_form.json",
    #     "img_prefix": "./data/coco_all/"
    # },
    # {
    #     "type": "OspreyLVISPosNeg",
    #     "ann_file": "./data/osprey-724k/osprey_lvis_positive_negative.json",
    #     "img_prefix": "./data/coco_all/"
    # },
]
