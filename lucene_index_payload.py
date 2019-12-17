from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceTokenizer
from org.apache.lucene.analysis.payloads import DelimitedPayloadTokenFilter, FloatEncoder, PayloadHelper
from org.apache.lucene.document import Document, Field, StringField, TextField, FieldType
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, Term
from org.apache.lucene.queries.payloads import PayloadScoreQuery, SumPayloadFunction
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.search.spans import SpanTermQuery, SpanBoostQuery
from org.apache.lucene.store import MMapDirectory
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.pylucene.search.similarities import PythonClassicSimilarity

import lucene


class PayloadSimilarity(PythonClassicSimilarity):
    def lengthNorm(self, numTerms):
        return 1.0

    def idf(self, docFreq, docCount):
        return 1.0

    def idfExplain(self, collectionStats, termStats):
        return Explanation.match(1.0, "inexplicable", [])

    def scorePayload(self, docId, start, end, payload):
        return PayloadHelper.decodeFloat(payload.bytes, payload.offset)


class PayloadAnalyzer(PythonAnalyzer):
    def __init__(self, encoder):
        super(PayloadAnalyzer, self).__init__()
        self.encoder = encoder

    def createComponents(self, field):
        source = WhitespaceTokenizer()
        filt = DelimitedPayloadTokenFilter(source, "|", self.encoder)
        return self.TokenStreamComponents(source, filt)

    def initReader(self, fieldName, reader):
        return reader


class LuceneIndex(object):

    def __init__(self, index_dir):
        BooleanQuery.setMaxClauseCount(2 ** 16)  # to avoid 'too many boolean clauses'

        self.index_dir = index_dir

        field_type = FieldType(TextField.TYPE_NOT_STORED)
        field_type.setOmitNorms(True)

        self.document = Document()
        self.fields = {
            'doc_id': Field('doc_id', '', StringField.TYPE_STORED),
            'f': Field('f', '', field_type)
        }

        for _, field in self.fields.items():
            self.document.add(field)

        self.directory = MMapDirectory(Paths.get(self.index_dir))
        self.analyzer = PayloadAnalyzer(FloatEncoder())
        self.similarity = PayloadSimilarity()
        self.payload_function = SumPayloadFunction()  # ???

        self.writer = None
        self.reader = None
        self.searcher = None

    def _init_writer(self):
        config = IndexWriterConfig(self.analyzer)
        config.setSimilarity(self.similarity)
        self.writer = IndexWriter(self.directory, config)

    def _init_searcher(self):
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(self.similarity)

    @staticmethod
    def _generate_document(x):
        terms = (f'{i}|{xi}' for i, xi in enumerate(x) if xi != 0)
        terms = ' '.join(terms)
        return terms

    def add(self, doc_id, feature):
        if self.writer is None:
            self._init_writer()

        text = self._generate_document(feature)

        self.fields['doc_id'].setStringValue(doc_id)
        self.fields['f'].setStringValue(text)

        self.writer.addDocument(self.document)
        # self.writer.commit()

    def _make_query(self, q):
        query = BooleanQuery.Builder()
        nonzero = ((i, qi) for (i, qi) in enumerate(q) if qi != 0)
        for i, qi in nonzero:
            term = Term('f', str(i))
            sub_query = SpanTermQuery(term)
            sub_query = SpanBoostQuery(sub_query, float(qi))  # boost = qi
            sub_query = PayloadScoreQuery(sub_query, self.payload_function, True)
            query.add(sub_query, BooleanClause.Occur.SHOULD)

        return query.build()

    def query(self, feature, limit=1000):
        if self.searcher is None:
            self._init_searcher()

        query = self._make_query(feature)
        scoreDocs = self.searcher.search(query, limit).scoreDocs
        return ((self.searcher.doc(result.doc)['doc_id'], result.score) for result in scoreDocs)

    def count(self):
        if self.searcher is None:
            try:
                self._init_searcher()
            except:
                return -1

        return self.reader.numDocs()

    def close(self):
        if self.writer is not None:
            self.writer.commit()
            self.writer.close()
            self.writer = None

        if self.searcher is not None:
            self.searcher = None
            self.reader.close()
            self.reader = None

    '''
    Magic methods to manage this object in a 'with' context.
    This assures that close() is called.
    '''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    import os
    import numpy as np
    from tqdm import tqdm

    nq, nx = 10, 100
    d = 2048
    q = np.random.rand(nq, d)
    x = np.random.rand(nx, d)

    p = .5
    sq = np.random.choice([True, False], size=(nq, d), p=[p, 1 - p])
    sx = np.random.choice([True, False], size=(nx, d), p=[p, 1 - p])

    q[sq] = 0
    x[sx] = 0

    scores = q.dot(x.T).squeeze()
    gt_ranks = np.sort(scores, axis=1)[:, ::-1]

    lucene_vm = lucene.initVM(vmargs=['-Djava.awt.headless=true'], initialheap='2g')
    lucene_vm.attachCurrentThread()
    with LuceneIndex('debug_index') as idx:
        for i, xi in enumerate(tqdm(x)):
            idx.add(str(i), xi)

    with LuceneIndex('debug_index') as idx:
        ranks = []
        for i, qi in enumerate(tqdm(q)):
            results = idx.query(qi)
            results = map(lambda x: x[1], results)
            results = list(results)
            ranks.append(results)

    ranks = np.array(ranks)
    correct = np.allclose(ranks, gt_ranks)
    if correct:
        os.system('rm -rf debug_index')
    else:
        print(ranks)
        print(gt_ranks)
