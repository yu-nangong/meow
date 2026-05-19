import os
import pandas as pd
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir, n_chunks=4):
        self.calendar = Calendar()
        self.h5dir = h5dir
        self.n_chunks = n_chunks
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    @staticmethod
    def _chunk_dates(dates, n_chunks):
        if not dates:
            return []
        n_chunks = min(n_chunks, len(dates))
        size = (len(dates) + n_chunks - 1) // n_chunks
        return [dates[i : i + size] for i in range(0, len(dates), size)]

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        chunks = self._chunk_dates(dates, self.n_chunks)
        log.inf(
            "Running model fitting on {} dates in {} chunks...".format(len(dates), len(chunks))
        )
        self.model.reset()
        for i, chunk in enumerate(chunks, 1):
            log.inf(
                "Fitting chunk {}/{}: {} dates ({} to {})".format(
                    i, len(chunks), len(chunk), chunk[0], chunk[-1]
                )
            )
            rawData = self.dloader.loadDates(chunk)
            xdf, ydf = self.featGenerator.genFeatures(rawData)
            del rawData
            self.model.partial_fit(xdf, ydf)
            del xdf, ydf
        self.model.finalize_fit()

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval_holdout(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        chunks = self._chunk_dates(dates, self.n_chunks)
        predParts = []
        for chunk in chunks:
            rawData = self.dloader.loadDates(chunk)
            xdf, ydf = self.featGenerator.genFeatures(rawData)
            del rawData
            ydf = ydf.copy()
            ydf.loc[:, "forecast"] = self.predict(xdf)
            del xdf
            predParts.append(ydf)
        return self.evaluator.metrics(pd.concat(predParts))

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        chunks = self._chunk_dates(dates, self.n_chunks)
        predParts = []
        for i, chunk in enumerate(chunks, 1):
            log.inf(
                "Eval chunk {}/{}: {} dates ({} to {})".format(
                    i, len(chunks), len(chunk), chunk[0], chunk[-1]
                )
            )
            rawData = self.dloader.loadDates(chunk)
            xdf, ydf = self.featGenerator.genFeatures(rawData)
            del rawData
            ydf = ydf.copy()
            ydf.loc[:, "forecast"] = self.predict(xdf)
            del xdf
            predParts.append(ydf)
        self.evaluator.eval(pd.concat(predParts))


if __name__ == "__main__":
    engine = MeowEngine(h5dir="data", cacheDir=None, n_chunks=8)
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
