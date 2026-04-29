#!/usr/bin/env python3
"""Test F1RadGraph con patch transformers 5.x"""

# Patch primero
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'encode_plus'):
    def encode_plus_wrapper(self, text, text_pair=None, add_special_tokens=True, 
                           padding=False, truncation=False, max_length=None,
                           stride=0, return_tensors=None, **kwargs):
        """Wrapper para compatibilidad con encode_plus deprecado."""
        return self.__call__(
            text, 
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs
        )
    PreTrainedTokenizerBase.encode_plus = encode_plus_wrapper
    print("✅ Patch encode_plus aplicado")

# Agregar build_inputs_with_special_tokens si no existe
if not hasattr(PreTrainedTokenizerBase, 'build_inputs_with_special_tokens'):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP] (Y [SEP])
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    print("✅ Patch build_inputs_with_special_tokens aplicado")

# Importar F1RadGraph
try:
    from radgraph import F1RadGraph
    print("✅ F1RadGraph importado")
    
    # Inicializar
    print("\n🔬 Inicializando F1RadGraph...")
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    print("✅ F1RadGraph inicializado correctamente!\n")
    
    # Test con ejemplo médico
    refs = ["no acute cardiopulmonary abnormality"]
    hyps = ["no acute cardiopulmonary abnormality"]
    
    print("🧪 Testing con:")
    print(f"  Reference: {refs[0]}")
    print(f"  Hypothesis: {hyps[0]}")
    print("\n⏳ Calculando scores...")
    
    mean_reward, reward_list, hyp_ann, ref_ann = f1radgraph(refs=refs, hyps=hyps)
    
    print("\n✅ Resultados:")
    print(f"  Mean reward: {mean_reward}")
    if isinstance(mean_reward, tuple):
        print(f"    RG_E (solo entidades): {mean_reward[0]:.4f}")
        print(f"    RG_ER (ent + rel): {mean_reward[1]:.4f}")
        print(f"    RG_bar_ER (completo): {mean_reward[2]:.4f}")
    
    print(f"\n  Reward list: {reward_list}")
    
    print("\n🎉 F1RadGraph funciona correctamente!")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error: {e}")
    print("\nTraceback completo:")
    traceback.print_exc()
