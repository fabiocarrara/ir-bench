ARG VERSION=latest
FROM python:$VERSION

RUN apt-get update \
    && apt-get install -y default-jdk ant

WORKDIR /usr/src/pylucene
RUN curl https://www.apache.org/dist/lucene/pylucene/pylucene-7.7.1-src.tar.gz \
    | tar -xz --strip-components=1
RUN cd jcc \
    && JCC_JDK=/usr/lib/jvm/default-java python setup.py install
ADD pylucene-7.7.1-scorePayload.patch .
RUN patch -s -p0 < pylucene-7.7.1-scorePayload.patch && rm pylucene-7.7.1-scorePayload.patch
RUN make all install JCC='python -m jcc' ANT=ant PYTHON=python NUM_FILES=8

RUN pip3 install --no-cache numpy tqdm dask[array] joblib scipy pandas h5py Pillow
WORKDIR ..
# RUN rm -rf pylucene
