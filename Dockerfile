FROM runpod/base:0.6.2-cuda12.2.0 as base

ENV PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig/:/usr/local/lib/pkgconfig/
# Update base and install build tools
RUN apt-get update

# --- Optional: System dependencies ---
COPY ./runpod-setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Python dependencies
COPY requirements.txt /requirements.txt
RUN PIP_REQUIRE_HASHES= python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-deps --upgrade blinker && \
    python3.11 -m pip install packaging && \
    python3.11 -m pip install torch>=2.4.0 && \
    PIP_REQUIRE_HASHES= python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt
    
RUN modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B

ADD . .

CMD python3.11 -u /runpod_handler.py