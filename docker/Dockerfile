FROM python:3.8

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    make \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf \
    /var/lib/apt/lists/* \
    /var/cache/apt/* \
    /usr/local/src/* \
    /tmp/*

RUN pip install --no-cache-dir\
    pandas\
    lightgbm \
    requests \
    "git+https://github.com/richmanbtc/crypto_data_fetcher.git@v0.0.15#egg=crypto_data_fetcher" \
    && rm -rf ~/.cache/pip

RUN cd /tmp \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -zxvf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && sudo make \
    && sudo make install \
    && cd ../ \
    && sudo rm -rf ta-lib-0.4.0-src.tar.gz \
    && sudo rm -rf ta-lib \
    && python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir install TA-Lib \
    && rm -rf /tmp/*

CMD ["/bin/bash"]