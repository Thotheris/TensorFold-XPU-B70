FROM ubuntu:26.04 AS toolchain
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl python3 python3-venv python3-dev gcc g++ libdrm2 libnuma1 ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*
COPY tools/xpu/debs.lock /opt/debs.lock
RUN mkdir /tmp/debs && while IFS='|' read -r package version url digest; do \
      file="/tmp/debs/${url##*/}"; \
      curl --fail --location --proto '=https' --tlsv1.2 --output "$file" "$url" && \
      printf '%s  %s\n' "$digest" "$file" | sha256sum --check - || exit 1; \
    done < /opt/debs.lock && dpkg -i /tmp/debs/*.deb && rm -rf /tmp/debs
COPY tools/xpu/constraints.txt /opt/constraints.txt
RUN python3 -m venv /opt/tf-venv
ENV PATH=/opt/tf-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pip install --constraint /opt/constraints.txt torch==2.14.1+xpu \
      --index-url https://download.pytorch.org/whl/xpu && \
    pip install --constraint /opt/constraints.txt \
      setuptools numpy huggingface-hub tokenizers safetensors jinja2 pytest && \
    pip check && pip freeze > /opt/tf-venv/freeze.txt

FROM toolchain AS serve
COPY . /opt/tensorfold
WORKDIR /opt/tensorfold
RUN pip install . --no-deps --no-build-isolation --constraint /opt/constraints.txt
ENTRYPOINT ["tensorfold"]
