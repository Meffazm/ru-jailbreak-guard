# Phase 8 — minimum YC infra for screenshot-only defense capture.
#
# Brings up: VPC + subnet + service account + IAM + zonal k8s cluster +
# 1-node group (4 vCPU / 16 GB). Approximately 250 RUB / day of runtime.
#
# Bring-up:
#   cd infra && terraform init && terraform apply
#   ../scripts/bootstrap_cloud.sh
#   # capture defense screenshots per docs/runbooks/cloud-defense-capture.md
#
# Tear-down:
#   ../scripts/teardown_cloud.sh
#
# All YC creds come from env vars: YC_TOKEN (OAuth), YC_CLOUD_ID, YC_FOLDER_ID,
# or from `yc config list` if the yc CLI is configured.

terraform {
  required_version = ">= 1.5"
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.131.0"
    }
  }
}

provider "yandex" {
  zone = "ru-central1-a"
  # token / cloud_id / folder_id read from env: YC_TOKEN / YC_CLOUD_ID / YC_FOLDER_ID
}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

variable "cluster_name" {
  type    = string
  default = "ru-jailbreak-guard"
}

variable "k8s_version" {
  type    = string
  default = "1.30"
}

variable "node_platform" {
  type        = string
  default     = "standard-v3"
  description = "YC compute platform; standard-v3 is the current Skylake/Cascade Lake."
}

variable "node_cores" {
  type    = number
  default = 4
}

variable "node_memory_gb" {
  type    = number
  default = 16
}

variable "node_disk_gb" {
  type    = number
  default = 64
}

# -----------------------------------------------------------------------------
# Network: VPC + subnet (single zone)
# -----------------------------------------------------------------------------

resource "yandex_vpc_network" "main" {
  name = "${var.cluster_name}-net"
}

resource "yandex_vpc_subnet" "main" {
  name           = "${var.cluster_name}-subnet"
  network_id     = yandex_vpc_network.main.id
  zone           = "ru-central1-a"
  v4_cidr_blocks = ["10.0.0.0/24"]
}

# -----------------------------------------------------------------------------
# Service account + IAM bindings (cluster needs editor perms in this folder)
# -----------------------------------------------------------------------------

resource "yandex_iam_service_account" "k8s" {
  name = "${var.cluster_name}-sa"
}

resource "yandex_resourcemanager_folder_iam_member" "editor" {
  folder_id = data.yandex_client_config.client.folder_id
  role      = "editor"
  member    = "serviceAccount:${yandex_iam_service_account.k8s.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "k8s_admin" {
  folder_id = data.yandex_client_config.client.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.k8s.id}"
}

data "yandex_client_config" "client" {}

# -----------------------------------------------------------------------------
# Managed k8s cluster (zonal master = free; pay only for nodes)
# -----------------------------------------------------------------------------

resource "yandex_kubernetes_cluster" "main" {
  name        = var.cluster_name
  network_id  = yandex_vpc_network.main.id
  description = "Phase 8 transient cluster for screenshot defense capture."

  master {
    version   = var.k8s_version
    public_ip = true
    zonal {
      zone      = "ru-central1-a"
      subnet_id = yandex_vpc_subnet.main.id
    }
  }

  service_account_id      = yandex_iam_service_account.k8s.id
  node_service_account_id = yandex_iam_service_account.k8s.id

  release_channel = "REGULAR"

  depends_on = [
    yandex_resourcemanager_folder_iam_member.editor,
    yandex_resourcemanager_folder_iam_member.k8s_admin,
  ]
}

# -----------------------------------------------------------------------------
# Single-node node group (4 vCPU / 16 GB / 64 GB disk) — fits the full stack.
# -----------------------------------------------------------------------------

resource "yandex_kubernetes_node_group" "workers" {
  cluster_id  = yandex_kubernetes_cluster.main.id
  name        = "${var.cluster_name}-workers"
  version     = var.k8s_version
  description = "Single 4 vCPU / 16 GB node — defense capture only."

  instance_template {
    platform_id = var.node_platform

    resources {
      cores  = var.node_cores
      memory = var.node_memory_gb
    }

    boot_disk {
      type = "network-ssd"
      size = var.node_disk_gb
    }

    network_interface {
      nat        = true
      subnet_ids = [yandex_vpc_subnet.main.id]
    }

    metadata = {
      ssh-keys = "ubuntu:${file("~/.ssh/id_ed25519.pub")}"
    }
  }

  scale_policy {
    fixed_scale {
      size = 1
    }
  }

  allocation_policy {
    location {
      zone = "ru-central1-a"
    }
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "cluster_id" {
  value = yandex_kubernetes_cluster.main.id
}

output "kubeconfig_command" {
  value       = "yc managed-kubernetes cluster get-credentials --id ${yandex_kubernetes_cluster.main.id} --external --force"
  description = "Run this to merge cluster credentials into ~/.kube/config."
}

output "node_group_id" {
  value = yandex_kubernetes_node_group.workers.id
}

output "approximate_daily_cost_rub" {
  value       = "~250 RUB/day for the ${var.node_cores}vCPU/${var.node_memory_gb}GB node + disk + master IPv4. Budget tight — destroy after capture."
  description = "Reminder string. Not a binding estimate."
}
