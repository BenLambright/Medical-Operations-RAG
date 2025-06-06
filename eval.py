import evaluate
from retrieval import Retriever
import json

class Eval():
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.llm = Retriever()
        self.rag_llm = self.llm.build_retriever_graph()
        self.row = []
        self.gold = []
        self.pred = []

    def process_data(self, path):
        with open(path, "r") as f:
            procedure = json.load(f)

        for item in procedure:
            print(f"processing {item["row"]}...")
            self.row.append(item["row"])
            self.gold.append(item["answer"])

            # inference
            inference = self.rag_llm.invoke({"question": item["question"]})
            self.pred.append(inference["answer"])

    def metrics(self):
        print("calculating metrics...")
        rouge_results = self.rouge.compute(predictions=self.pred, references=self.gold)
        bleu_results = self.bleu.compute(references=self.gold, predictions=self.pred)

        return rouge_results, bleu_results

print("setting up the model...")
eval_obj = Eval()
eval_obj.process_data("data/materials.json")
# eval_obj.process_data("data/operation.json")
# eval_obj.process_data("data/materials.json")
print("")
print(eval_obj.pred)
print(eval_obj.metrics())


# example
# rouge_results = rouge.compute(
#     predictions=["The operation performed was a left partial nephrectomy. Key materials included a Bovie, Finochietto retractor, bulldog clamp, Vicryl sutures, Surgicel, and a 19-French Blake drain. The procedure involved a flank incision, kidney mobilization, tumor dissection with argon beam coagulation, and closure with sutures and metallic clips."],
#     references=["Epidural anesthesia was administered in the holding area, after which the patient was transferred into the operating room.  General endotracheal anesthesia was administered, after which the patient was positioned in the flank standard position.  A left flank incision was made over the area of the twelfth rib.  The subcutaneous space was opened by using the Bovie.  The ribs were palpated clearly and the fascia overlying the intercostal space between the eleventh and twelfth rib was opened by using the Bovie.  The fascial layer covering of the intercostal space was opened completely until the retroperitoneum was entered.  Once the retroperitoneum had been entered, the incision was extended until the peritoneal envelope could be identified.  The peritoneum was swept medially.  The Finochietto retractor was then placed for exposure.  The kidney was readily identified and was mobilized from outside Gerota's fascia.  The ureter was dissected out easily and was separated with a vessel loop.  The superior aspect of the kidney was mobilized from the superior attachment.  The pedicle of the left kidney was completely dissected revealing the vein and the artery.  The artery was a single artery and was dissected easily by using a right-angle clamp.  A vessel loop was placed around the renal artery.  The tumor could be easily palpated in the lateral lower pole to mid pole of the left kidney.  The Gerota's fascia overlying that portion of the kidney was opened in the area circumferential to the tumor.  Once the renal capsule had been identified, the capsule was scored using a Bovie about 0.5 cm lateral to the border of the tumor.  Bulldog clamp was then placed on the renal artery.  The tumor was then bluntly dissected off of the kidney with a thin rim of a normal renal cortex.  This was performed by using the blunted end of the scalpel.  The tumor was removed easily.  The argon beam coagulation device was then utilized to coagulate the base of the resection.  The visible larger bleeding vessels were oversewn by using 4-0 Vicryl suture.  The edges of the kidney were then reapproximated by using 2-0 Vicryl suture with pledgets at the ends of the sutures to prevent the sutures from pulling through.  Two horizontal mattress sutures were placed and were tied down.  The Gerota's fascia was then also closed by using 2-0 Vicryl suture.  The area of the kidney at the base was covered with Surgicel prior to tying the sutures.  The bulldog clamp was removed and perfect hemostasis was evident.  There was no evidence of violation into the calyceal system.  A 19-French Blake drain was placed in the inferior aspect of the kidney exiting the left flank inferior to the incision.  The drain was anchored by using silk sutures.  The flank fascial layers were closed in three separate layers in the more medial aspect.  The lateral posterior aspect was closed in two separate layers using Vicryl sutures."]
# )
#
# print(rouge_results)
#
# bleu_results = bleu.compute(
#     predictions=["The operation performed was a left partial nephrectomy. Key materials included a Bovie, Finochietto retractor, bulldog clamp, Vicryl sutures, Surgicel, and a 19-French Blake drain. The procedure involved a flank incision, kidney mobilization, tumor dissection with argon beam coagulation, and closure with sutures and metallic clips."],
#     references=["Epidural anesthesia was administered in the holding area, after which the patient was transferred into the operating room.  General endotracheal anesthesia was administered, after which the patient was positioned in the flank standard position.  A left flank incision was made over the area of the twelfth rib.  The subcutaneous space was opened by using the Bovie.  The ribs were palpated clearly and the fascia overlying the intercostal space between the eleventh and twelfth rib was opened by using the Bovie.  The fascial layer covering of the intercostal space was opened completely until the retroperitoneum was entered.  Once the retroperitoneum had been entered, the incision was extended until the peritoneal envelope could be identified.  The peritoneum was swept medially.  The Finochietto retractor was then placed for exposure.  The kidney was readily identified and was mobilized from outside Gerota's fascia.  The ureter was dissected out easily and was separated with a vessel loop.  The superior aspect of the kidney was mobilized from the superior attachment.  The pedicle of the left kidney was completely dissected revealing the vein and the artery.  The artery was a single artery and was dissected easily by using a right-angle clamp.  A vessel loop was placed around the renal artery.  The tumor could be easily palpated in the lateral lower pole to mid pole of the left kidney.  The Gerota's fascia overlying that portion of the kidney was opened in the area circumferential to the tumor.  Once the renal capsule had been identified, the capsule was scored using a Bovie about 0.5 cm lateral to the border of the tumor.  Bulldog clamp was then placed on the renal artery.  The tumor was then bluntly dissected off of the kidney with a thin rim of a normal renal cortex.  This was performed by using the blunted end of the scalpel.  The tumor was removed easily.  The argon beam coagulation device was then utilized to coagulate the base of the resection.  The visible larger bleeding vessels were oversewn by using 4-0 Vicryl suture.  The edges of the kidney were then reapproximated by using 2-0 Vicryl suture with pledgets at the ends of the sutures to prevent the sutures from pulling through.  Two horizontal mattress sutures were placed and were tied down.  The Gerota's fascia was then also closed by using 2-0 Vicryl suture.  The area of the kidney at the base was covered with Surgicel prior to tying the sutures.  The bulldog clamp was removed and perfect hemostasis was evident.  There was no evidence of violation into the calyceal system.  A 19-French Blake drain was placed in the inferior aspect of the kidney exiting the left flank inferior to the incision.  The drain was anchored by using silk sutures.  The flank fascial layers were closed in three separate layers in the more medial aspect.  The lateral posterior aspect was closed in two separate layers using Vicryl sutures."]
# )
#
# print(bleu_results)