import os
import argparse
import jsonlines
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from tokenizers import BertWordPieceTokenizer

from dataset.mrc_ner_dataset import MRCNERDataset
from train.mrc_ner_trainer import BertLabeling
from utils.get_parser import get_parser
from metrics.functional.query_span_f1 import extract_flat_spans, extract_nested_spans


def yd_add_parser(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--inference_dir", type=str, required=True)
    parser.add_argument("--inference_output_dir", type=str, required=True)
    parser.add_argument(
        "--att_list",
        required=True,
        # 33att
        # default="['ActiveIngredients','AgeRangeDescription','BatteryCellComposition','Brand','CaffeineContent','CapacityUnit','CoffeeRoastType','Color','DietType','DosageForm','EnergyUnit','FinishType','Flavor','FormulationType','HairType','Ingredients','ItemForm','ItemShape','LiquidContentsDescription','Material','MaterialFeature','MaterialTypeFree','PackageSizeName','Pattern','PatternType','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','TeaVariety','Variety']",
        type=eval,
    )
    parser.add_argument("--dataset_sign", type=str, choices=["ontonotes4", "msra", "conll03", "ace04", "ace05","az15","az33","ae48"], default="conll03")
    return parser


def get_query_index_to_label_cate(dataset_sign):
    # NOTICE: need change if you use other datasets.
    # please notice it should in line with the mrc-ner.test/train/dev json file
    if dataset_sign == "conll03":
        return {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}
    elif dataset_sign == "ace04":
        return {1: "GPE", 2: "ORG", 3: "PER", 4: "FAC", 5: "VEH", 6: "LOC", 7: "WEA"}
    elif dataset_sign == "az33":
        return {0: 'ActiveIngredients', 1: 'AgeRangeDescription', 2: 'BatteryCellComposition', 3: 'Brand', 4: 'CaffeineContent', 5: 'CapacityUnit', 6: 'CoffeeRoastType', 7: 'Color', 8: 'DietType', 9: 'DosageForm', 10: 'EnergyUnit', 11: 'FinishType', 12: 'Flavor', 13: 'FormulationType', 14: 'HairType', 15: 'Ingredients', 16: 'ItemForm', 17: 'ItemShape', 18: 'LiquidContentsDescription', 19: 'Material', 20: 'MaterialFeature', 21: 'MaterialTypeFree', 22: 'PackageSizeName', 23: 'Pattern', 24: 'PatternType', 25: 'ProductBenefit', 26: 'Scent', 27: 'SkinTone', 28: 'SkinType', 29: 'SpecialIngredients', 30: 'TargetGender', 31: 'TeaVariety', 32: 'Variety'}
    elif dataset_sign == "az15":
        return {0: 'ActiveIngredients', 1: 'AgeRangeDescription', 2: 'Color', 3: 'FinishType', 4: 'Flavor', 5: 'HairType', 6: 'ItemForm', 7: 'Material', 8: 'ProductBenefit', 9: 'Scent', 10: 'SkinTone', 11: 'SkinType', 12: 'SpecialIngredients', 13: 'TargetGender', 14: 'Variety'}
    elif dataset_sign == "ae48":
        return {0: 'ApplicablePlace', 1: 'AthleticShoeType', 2: 'BackSideMaterial', 3: 'BodyMaterial', 4: 'BrandName', 5: 'Capacity', 6: 'Category', 7: 'ClosureType', 8: 'Collar', 9: 'Color', 10: 'DepartmentName', 11: 'DerivativeSeries', 12: 'FabricType', 13: 'Feature', 14: 'FingerboardMaterial', 15: 'Fit', 16: 'Function', 17: 'Gender', 18: 'HoseHeight', 19: 'InsoleMaterial', 20: 'IsCustomized', 21: 'ItemType', 22: 'Length', 23: 'LensesOpticalAttribute', 24: 'LevelOfPractice', 25: 'LiningMaterial', 26: 'Material', 27: 'Model', 28: 'ModelNumber', 29: 'Name', 30: 'OuterwearType', 31: 'OutsoleMaterial', 32: 'PatternType', 33: 'ProductType', 34: 'Season', 35: 'Size', 36: 'SleeveLengthCm', 37: 'SportType', 38: 'SportsType', 39: 'StrapType', 40: 'Style', 41: 'Technology', 42: 'Type', 43: 'TypeOfSports', 44: 'UpperHeight', 45: 'UpperMaterial', 46: 'Voltage', 47: 'Weight'}


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    parser = yd_add_parser(parser)
    args = parser.parse_args()
    os.makedirs(args.inference_output_dir, exist_ok=True)

    trained_mrc_ner_model = BertLabeling(args)
    args.is_chinese = False
    args.flat_ner = True

    if args.pretrained_checkpoint:
        print("yd: start to load ckpt !")
        trained_mrc_ner_model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])
        print("yd: finish loading ckpt !")


    # obtain tokenizer
    vocab_path = os.path.join(args.bert_config_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token
    data_tokenizer = BertWordPieceTokenizer(vocab_path)

    for att in args.att_list:
        inference_file = os.path.join(args.inference_dir, att + '_mrc.gold')
        inference_output_file = os.path.join(args.inference_output_dir, att + '.jsonl')

        if not os.path.isfile(inference_file):
            continue
        # obtain dataset and dataloader
        dataset = MRCNERDataset(
            json_path=inference_file,
            tokenizer=data_tokenizer,
            max_length=args.max_length,
            is_chinese=args.is_chinese,
            pad_to_maxlen=False,
        )

        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        # obtain query2label_dict (used by NER classes)
        query2label_dict = get_query_index_to_label_cate(args.dataset_sign)

        # **YD** get model into gpu
        trained_mrc_ner_model.model.cuda()

        output = []
        for index, batch in enumerate(tqdm(data_loader)):
            # if index < 256:
            #     continue
            # print(f'index: {index}')
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
            attention_mask = (tokens != 0).long()

            gpu_tokens = tokens.cuda()
            gpu_attention_mask = attention_mask.cuda()
            gpu_token_type_ids = token_type_ids.cuda()

            start_logits, end_logits, span_logits = trained_mrc_ner_model.model(
                gpu_tokens, attention_mask=gpu_attention_mask,token_type_ids=gpu_token_type_ids
            )
            start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0
            start_preds = start_preds.long().cpu()
            end_preds = end_preds.long().cpu()
            span_preds = span_preds.long().cpu()

            subtokens_idx_lst = tokens.numpy().tolist()[0]
            subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst]
            label_cate = query2label_dict[label_idx.item()]
            readable_input_str = data_tokenizer.decode(subtokens_idx_lst, skip_special_tokens=True)

            # **YD** avoid selecting spans from questions.
            sub_token_type_ids = token_type_ids[0]
            assert len(sub_token_type_ids) == len(subtokens_lst)

            if args.flat_ner:
                # print(f'label_cate: {label_cate}')
                # print(f'start_preds:{start_preds.shape}; {start_preds}')
                # print(f'end_preds:{end_preds.shape} {end_preds}')
                # print(f'span_preds:{span_preds.shape} {span_preds}')
                # print(f'attention_mask: {attention_mask}')

                entities_info = extract_flat_spans(torch.squeeze(start_preds), torch.squeeze(end_preds),
                                                   torch.squeeze(span_preds), torch.squeeze(attention_mask),
                                                   pseudo_tag=label_cate)
                entity_lst = []

                if len(entities_info) != 0:
                    for entity_info in entities_info:
                        start, end = entity_info[0], entity_info[1]
                        # **YD** avoid selecting spans from questions.
                        if sub_token_type_ids[start] == 0 or sub_token_type_ids[end] == 0:
                            continue

                        entity_string = " ".join(subtokens_lst[start: end])
                        entity_string = entity_string.replace(" ##", "")
                        entity_lst.append((start, end, entity_string, entity_info[2]))

            else:
                match_preds = span_logits > 0
                entities_info = extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask,
                                                     pseudo_tag=label_cate)

                entity_lst = []

                if len(entities_info) != 0:
                    for entity_info in entities_info:
                        start, end = entity_info[0], entity_info[1]
                        # **YD** avoid selecting spans from questions.
                        if sub_token_type_ids[start] == 0 or sub_token_type_ids[end] == 0:
                            continue
                        entity_string = " ".join(subtokens_lst[start: end + 1])
                        entity_string = entity_string.replace(" ##", "")
                        entity_lst.append((start, end + 1, entity_string, entity_info[2]))

            # if len(entity_lst) > 0:
            #     print(f'entity_lst: {entity_lst}')
            #     import sys
            #     sys.exit()

            ori_instance = dataset.all_data[index]
            output_instance = {
                'asin': ori_instance.get('asin', ''),
                'attribute': att,
                'product_type': ori_instance.get('product_type', ''),
                'sentence': ori_instance.get('context', ''),
                'pred_result': [entity_l[2] for entity_l in entity_lst]
            }
            output.append(output_instance)

        with jsonlines.open(inference_output_file, 'w') as writer:
            writer.write_all(output)
        # print("*=" * 10)
        # print(f"Given input: {readable_input_str}")
        # print(f"Model predict: {entity_lst}")
        # entity_lst is a list of (subtoken_start_pos, subtoken_end_pos, substring, entity_type)


if __name__ == '__main__':
    main()
