class Prompter:
    def __init__(self):
        self.generator_template = "Dưới đây là một Instruction mô tả nhiệm vụ. Viết một Response hoàn thành yêu cầu một cách thích hợp.\n\n ### Instruction:\n{instruction}\n\n### Response: Hãy suy nghĩ từng bước.\n"
        self.verifier_template = "### Câu hỏi: {question}\n### Trả lời: {answer}"
        
    def generator_prompt(
        self,
        instruction: str,
        response: str = None,
    ) -> str:
        
        prompt = self.generator_template.format(instruction = instruction)
        if response:
            prompt = f"{prompt}{response}"
        return prompt
    
    def verifier_prompt(
        self,
        question: str,
        answer: str,
    ) -> str:
        return self.verifier_template.format(question = question, answer = answer)
    
    def get_response(self, output: str) -> str:
        parts = output.split("### Response: Hãy suy nghĩ từng bước.")
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return ""