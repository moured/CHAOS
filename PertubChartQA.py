class PertubChartQA(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'CHAOS_all': 'https://huggingface.co/datasets/omoured/CHAOS/blob/main/CHAOS_all.tsv',
        'CHAOS_vision':'https://huggingface.co/datasets/omoured/CHAOS/blob/main/CHAOS_vision.tsv',
        'CHAOS_text': 'https://huggingface.co/datasets/omoured/CHAOS/blob/main/CHAOS_text.tsv',
    }

    DATASET_MD5 = {
        'CHAOS_all': 'b87c3ac6bd9a92f5d079fec92ef213e9',
        'CHAOS_vision':'ba558b95f25ee73df736c62a8b1eb6ac',
        'CHAOS_text': 'fd75a0fb2140b7ca929160197571ad58',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs
        
    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # If model name starts with "Janus", remove trailing dot from each prediction
        data['prediction'] = [x.rstrip('.') if x.endswith('.') else x for x in data['prediction']]

        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        hit = hit_calculate(res, dataset)
        ret = dict()

        splits = set(data['split'])
        for sp in splits:
            sub = [r for l, r in zip(lines, res) if l['split'] == sp]
            hit = hit_calculate(sub, dataset)
            ret[sp] = np.mean(hit) * 100
        sub = [r for l, r in zip(lines, res)]
        hit = hit_calculate(sub, dataset)
        # ret['Overall'] = np.mean(hit) * 100

        # --- Start of New Code to Compute "overall_xx" metrics ---
        import re
        from collections import defaultdict

        vision_perturbations = [
            "blotches", "color", "dilation", "elastic_transform", "erosion",
            "fibrous_noise", "gaussian_blur", "motion_blur", "shifting", "uneven_brightness"
        ]
        text_perturbations = [
            "AddChar", "DeleteChar", "ReplaceChar", "SwapChar", "WordReplacement"
        ]

        perturbation_level_scores = defaultdict(list)
        vision_level_scores = defaultdict(list)
        text_level_scores = defaultdict(list)

        for sp in splits:
            match = re.match(r"([^_]+)_(easy|middle|difficult)_(human|augmented)", sp)
            if not match:
                continue
            perturbation, level, _ = match.groups()

            key = f"{perturbation}_{level}"
            if sp in ret:
                perturbation_level_scores[key].append(ret[sp])

            if perturbation in vision_perturbations:
                vision_level_scores[level].append(ret[sp])
            elif perturbation in text_perturbations:
                text_level_scores[level].append(ret[sp])

        for key, scores in perturbation_level_scores.items():
            ret[f"overall_{key}"] = np.mean(scores)

        for level, scores in vision_level_scores.items():
            ret[f"overall_vision_{level}"] = np.mean(scores)

        for level, scores in text_level_scores.items():
            ret[f"overall_text_{level}"] = np.mean(scores)
        # --- End of New Code ---

        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret