"""
任务二：MLLM 多模态科学问答 (ScienceQA)
使用 Qwen2-VL-7B-Instruct 模型进行多模态问答评测
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import re
import random
from tqdm import tqdm
import json
from typing import Dict, List
from PIL import Image
from qwen_vl_utils import process_vision_info
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False


class ScienceQAEvaluator:
    """ScienceQA 多模态问答评测器"""
    
    def __init__(self, model_name: str = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {self.device}")
        print(f"4-bit量化: {use_4bit}")
        
        # Load processor with use_fast=False to avoid compatibility issues
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print("模型加载完成！")
    
    def load_dataset(self, num_samples: int = 50, seed: int = 42) -> List[Dict]:
        print(f"正在加载ScienceQA数据集...")
        dataset = load_dataset("derek-thomas/ScienceQA", split="test")
        
        samples_with_image = []
        for sample in dataset:
            if sample.get('image') is not None:
                samples_with_image.append(sample)
        
        print(f"找到 {len(samples_with_image)} 条包含图片的样本")
        
        random.seed(seed)
        
        if len(samples_with_image) < num_samples:
            print(f"警告：可用样本数 ({len(samples_with_image)}) 少于请求数 ({num_samples})")
            selected_samples = samples_with_image
        else:
            selected_samples = random.sample(samples_with_image, num_samples)
        
        print(f"成功加载 {len(selected_samples)} 条测试数据")
        return selected_samples
    
    def format_choices(self, choices: List[str]) -> str:
        options = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted = []
        for i, choice in enumerate(choices):
            if i < len(options):
                formatted.append(f"{options[i]}. {choice}")
        return "\n".join(formatted)
    
    def extract_choice(self, text: str, num_choices: int) -> str:
        valid_options = ['A', 'B', 'C', 'D', 'E', 'F'][:num_choices]
        
        patterns = [
            r'答案是[：:]\s*([A-F])',
            r'选择[：:]\s*([A-F])',
            r'答案[：:]\s*([A-F])',
            r'选项\s*([A-F])',
            r'^([A-F])[\.。]',
            r'[^A-Z]([A-F])[^A-Z]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                option = match.upper()
                if option in valid_options:
                    return option
        
        for char in text.upper():
            if char in valid_options:
                return char
        
        return 'A'
    
    def create_prompt(self, question: str, choices: List[str], hint: str = None) -> str:
        formatted_choices = self.format_choices(choices)
        
        prompt = f"""请根据图片回答以下科学问题。请仔细观察图片，分析问题，并从给定的选项中选择正确答案。

问题：{question}

选项：
{formatted_choices}
"""
        
        if hint:
            prompt += f"\n提示：{hint}\n"
        
        prompt += "\n请直接给出你认为正确的选项字母（A/B/C/D等），并简要说明理由。\n答案是："
        
        return prompt
    
    def generate_answer(self, image: Image.Image, question: str, choices: List[str], 
                       hint: str = None, max_new_tokens: int = 256) -> str:
        prompt = self.create_prompt(question, choices, hint)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        
        generated_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return generated_text
    
    def evaluate(self, samples: List[Dict], save_results: bool = True) -> Dict:
        correct = 0
        total = len(samples)
        results = []
        
        print(f"\n开始评估 {total} 条样本...")
        
        for i, sample in enumerate(tqdm(samples, desc="评估进度")):
            try:
                image = sample['image']
                question = sample['question']
                choices = sample['choices']
                answer_idx = sample['answer']
                hint = sample.get('hint', None)
                
                options = ['A', 'B', 'C', 'D', 'E', 'F']
                ground_truth = options[answer_idx]
                
                generated = self.generate_answer(image, question, choices, hint)
                predicted = self.extract_choice(generated, len(choices))
                
                is_correct = (predicted == ground_truth)
                
                if is_correct:
                    correct += 1
                
                result = {
                    "index": i,
                    "question": question,
                    "choices": choices,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "generated_text": generated,
                    "correct": is_correct
                }
                results.append(result)
                
            except Exception as e:
                print(f"\n样本 {i} 处理出错: {str(e)}")
                result = {
                    "index": i,
                    "question": sample.get('question', 'N/A'),
                    "ground_truth": "N/A",
                    "predicted": "ERROR",
                    "generated_text": str(e),
                    "correct": False
                }
                results.append(result)
        
        accuracy = correct / total if total > 0 else 0
        
        if save_results:
            output = {
                "model": self.model_name,
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "results": results
            }
            
            with open("task2_results.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到 task2_results.json")

            # plot evaluation results and save figure
            try:
                incorrect = total - correct
                labels = ["Correct", "Incorrect"]
                counts = [correct, incorrect]
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(labels, counts, color=["#4CAF50", "#F44336"])
                ax.set_ylabel("Count")
                ax.set_title("ScienceQA Evaluation Results")
                ax.set_ylim(0, max(counts) * 1.1 if counts else 1.0)

                for bar, value in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        str(value),
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()
                plt.savefig("task2_accuracy.png", dpi=300)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to save figure: {e}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }


def main():
    evaluator = ScienceQAEvaluator(
        model_name="/root/autodl-tmp/models/Qwen2-VL-7B-Instruct",
        use_4bit=True
    )
    
    samples = evaluator.load_dataset(num_samples=50, seed=42)
    results = evaluator.evaluate(samples, save_results=True)
    
    print("\n" + "="*50)
    print("评估完成！")
    print("="*50)
    print(f"模型: {evaluator.model_name}")
    print(f"总样本数: {results['total']}")
    print(f"正确数量: {results['correct']}")
    print(f"准确率: {results['accuracy']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
