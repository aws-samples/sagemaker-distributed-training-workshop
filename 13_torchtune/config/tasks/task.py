from lm_eval.tasks.scrolls.task import _SCROLLSSummaryTask, _SCROLLSTask, _download_metric, load_metric

class Samsum(_SCROLLSSummaryTask):
    DATASET_PATH = "Samsung/samsum"
    DATASET_NAME = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = load_metric(_download_metric(), config_name="qmsum")

    def has_training_docs(self):
        return False

    def _process_doc(self, doc):
        doc["summary"] = [doc["summary"]]
        return [doc]

    def doc_to_target(self, doc):
        return f" {doc['summary']}"

    def doc_to_text(self, doc):
        doc = f"{doc['dialogue']}\n\nQuestion: What is a summary of the preceding text?\nAnswer:"
        #print(f"{doc=}")
        return doc

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["summary"]),
            "rouge2": (results[0], doc["summary"]),
            "rougeL": (results[0], doc["summary"]),
        }

    def download(self, *args, **kwargs):
        super(_SCROLLSTask, self).download(*args, **kwargs)
        del self.dataset["test"]
        del self.dataset["train"]
        for split in self.dataset:
            self.dataset[split] = self.dataset[split]