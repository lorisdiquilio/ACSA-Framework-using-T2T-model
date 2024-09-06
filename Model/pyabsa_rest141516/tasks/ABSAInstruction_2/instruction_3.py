class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class CategoryInstruction(Instruction):
    def __init__(self, bos_instruction=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        if self.bos_instruction is None:
            self.bos_instruction = f"""
Definition: The input are sentences about a product. The task is to extract the categories and their corresponding polarities only if the sentence is related to the the packaging. Here are some examples:
    
example 1-
input: Only complaint is the pump action container. As you near the end of bottle, pumping no longer works and there will still be a lot left in there. To not waste the remaining lotion, you have to dig it out with your finger, which is a pain.
{self.eos_instruction}
GENERAL#SATISFACTION-DESIGN:Negative|TECHNICAL#SUCTION_ABILITY:Negative|INTEGRITY#LEFTOVER:Negative

example 2-
input: Initially, I struggled with the bottle's tip; it's incredibly difficult to dispense anything, especially when aiming for the eyes. With painful arthritis in my hands, the short tip posed a significant challenge.
{self.eos_instruction}
GENERAL#USABILITY:Negative|GENERAL#SIZE:Negative

Now extract categories:polarities for the following example:
input: """
        if self.eos_instruction is None:
            self.eos_instruction = "\nlet us extract categories:polarities one by one: \n"

        if not self.bos_instruction:
            self.bos_instruction = bos_instruction
        if not self.eos_instruction:
            self.eos_instruction = eos_instruction

    def prepare_input(self, input_text, categories="", polarities=""):
        return (
            self.bos_instruction
            + input_text
            + f"The categories are: {categories}"
            + f"The polarities are: {polarities}"
            + self.eos_instruction
        )


"""Definition: The input are sentences about a product. The task is to extract the categories and their corresponding polarities only if the sentence is related to the the packaging. Here are some examples:
    
example 1-
input: Only complaint is the pump action container. As you near the end of bottle, pumping no longer works and there will still be a lot left in there. To not waste the remaining lotion, you have to dig it out with your finger, which is a pain.
{self.eos_instruction}
GENERAL#SATISFACTION-DESIGN:Negative|TECHNICAL#SUCTION_ABILITY:Negative|INTEGRITY#LEFTOVER:Negative

example 2-
input: Initially, I struggled with the bottle's tip; it's incredibly difficult to dispense anything, especially when aiming for the eyes. With painful arthritis in my hands, the short tip posed a significant challenge.
{self.eos_instruction}
GENERAL#USABILITY:Negative|GENERAL#SIZE:Negative


"""


'''Definition: The input are sentences about a product. The task is to extract the categories and their corresponding polarities in the review. Here are some examples:
    
example 1-
input: I haven’t seen any significant changes in my wrinkles or fine lines. I was hoping for better results considering the price of the product.
{self.eos_instruction}
Anti-Aging:Negative|Results:Negative|Price/Value:Negative

example 2-
input: I like it, the smell is good, it moisturised hair, they shine after.
{self.eos_instruction}
Satisfaction:Positive|Smell:Positive|Hydration:Positive|Tone:Positive

example 3-
input: While the product works well, it doesn’t last as long as I’d like. The quantity seems small considering the price point.
{self.eos_instruction}
Results:Positive|Durability:Negative|Size:Negative|Price/Value:Negative

'''