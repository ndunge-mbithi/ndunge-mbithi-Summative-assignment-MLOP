# nginx load balancer. Default: official nginx image. BASE_IMAGE may instead be a
# Debian/Ubuntu image, in which case nginx is installed from the distro archive.
ARG BASE_IMAGE=nginx:1.27-alpine
FROM ${BASE_IMAGE}
RUN command -v nginx >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*)
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
