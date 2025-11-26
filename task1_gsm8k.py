"""
任务一：LLM 数学推理评测 (GSM8K)
使用 Qwen2.5-7B-Instruct 模型进行数学推理评测
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import random
from tqdm import tqdm
import json
from typing import Dict, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False


class GSM8KEvaluator:
    """GSM8K 数学推理评测器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {self.device}")
        print(f"4-bit量化: {use_4bit}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print("模型加载完成！")
    
    def load_dataset(self, num_samples: int = 50, seed: int = 42) -> List[Dict]:
        print(f"正在加载GSM8K数据集...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[i] for i in indices]
        
        print(f"成功加载 {len(samples)} 条测试数据")
        return samples
    
    def extract_answer(self, text: str) -> str:
        """从文本中提取数值答案"""
        patterns = [
            r'####\s*([+-]?[\d,]+\.?\d*)',
            r'答案是[：:]\s*([+-]?[\d,]+\.?\d*)',
            r'答案为[：:]\s*([+-]?[\d,]+\.?\d*)',
            r'最终答案[：:]\s*([+-]?[\d,]+\.?\d*)',
            r'=\s*([+-]?[\d,]+\.?\d*)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '')
        
        numbers = re.findall(r'([+-]?[\d,]+\.?\d*)', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def normalize_answer(self, answer: str) -> float:
        try:
            return float(answer.replace(',', ''))
        except:
            return float('inf')
    
    def create_prompt(self, question: str) -> str:
        prompt = f"""请解决以下数学问题，并给出详细的推理过程。最后请用"答案是："的格式给出最终答案（1个数字）。

问题：{question}

请一步步思考并解答"""
        return prompt
    
    def generate_answer(self, question: str, max_new_tokens: int = 512) -> str:
        prompt = self.create_prompt(question)
        
        messages = [
            {"role": "system", "content": "你是一个数学专家，擅长解决数学问题。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def evaluate(self, samples: List[Dict], save_results: bool = True) -> Dict:
        correct = 0
        total = len(samples)
        results = []
        
        print(f"\n开始评估 {total} 条样本...")
        
        for i, sample in enumerate(tqdm(samples, desc="评估进度")):
            question = sample['question']
            ground_truth = sample['answer']
            
            gt_answer = self.extract_answer(ground_truth)
            gt_value = self.normalize_answer(gt_answer)
            
            try:
                generated = self.generate_answer(question)
                pred_answer = self.extract_answer(generated)
                pred_value = self.normalize_answer(pred_answer)
                
                is_correct = abs(pred_value - gt_value) < 1e-2
                
                if is_correct:
                    correct += 1
                
                result = {
                    "index": i,
                    "question": question,
                    "ground_truth": gt_answer,
                    "predicted": pred_answer,
                    "generated_text": generated,
                    "correct": is_correct
                }
                results.append(result)
                
            except Exception as e:
                print(f"\n样本 {i} 处理出错: {str(e)}")
                result = {
                    "index": i,
                    "question": question,
                    "ground_truth": gt_answer,
                    "predicted": "ERROR",
                    "generated_text": str(e),
                    "correct": False
                }
                results.append(result)
        
        accuracy = correct / total
        
        if save_results:
            output = {
                "model": self.model_name,
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "results": results
            }
            
            with open("task1_results.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到 task1_results.json")

            # plot evaluation results and save figure
            try:
                incorrect = total - correct
                labels = ["Correct", "Incorrect"]
                counts = [correct, incorrect]
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(labels, counts, color=["#4CAF50", "#F44336"])
                ax.set_ylabel("Count")
                ax.set_title("GSM8K Evaluation Results")
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
                plt.savefig("task1_accuracy.png", dpi=300)
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
    evaluator = GSM8KEvaluator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_4bit=True
    )
    
    samples = evaluator.load_dataset(num_samples=50, seed=20)
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
