# Yandex Cloud infrastructure: VPC + subnet + service account + IAM + zonal
# managed k8s cluster + single node group. Bring-up via `make cloud-up`,
# tear-down via `make cloud-down`.

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
  token     = var.yc_token
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = "ru-central1-a"
}

variable "yc_token" {
  type        = string
  sensitive   = true
  default     = null
  description = "Yandex Cloud IAM/OAuth token. Falls back to YC_TOKEN env var if null."
}

variable "yc_cloud_id" {
  type        = string
  default     = null
  description = "Yandex Cloud ID. Falls back to YC_CLOUD_ID env var if null."
}

variable "yc_folder_id" {
  type        = string
  default     = null
  description = "Yandex Cloud folder ID. Falls back to YC_FOLDER_ID env var if null."
}

variable "cluster_name" {
  type    = string
  default = "ru-jailbreak-guard"
}

variable "k8s_version" {
  type    = string
  default = "1.33"
  # `make cloud-up` overrides this with the latest REGULAR-channel version
  # discovered via `yc managed-kubernetes list-versions`.
}

variable "node_platform" {
  type        = string
  default     = "standard-v3"
  description = "YC compute platform; standard-v3 is the current Skylake/Cascade Lake."
}

variable "node_cores" {
  type    = number
  default = 8
  # 8 vCPU / 32 GB / 128 GB sized for concurrent image pulls of the full stack
  # (predictors + UI + monitoring ≈ 20 GB). Smaller nodes hit disk-pressure.
}

variable "node_memory_gb" {
  type    = number
  default = 32
}

variable "node_disk_gb" {
  type    = number
  default = 128
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
  description = "Managed k8s cluster for ru-jailbreak-guard."

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
# Single-node node group sized to fit the full stack (predictors + UI +
# monitoring) without disk-pressure during concurrent image pulls.
# -----------------------------------------------------------------------------

resource "yandex_kubernetes_node_group" "workers" {
  cluster_id  = yandex_kubernetes_cluster.main.id
  name        = "${var.cluster_name}-workers"
  version     = var.k8s_version
  description = "Worker node group for ru-jailbreak-guard."

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
  value       = "~250 RUB/day for the ${var.node_cores}vCPU/${var.node_memory_gb}GB node + disk + master IPv4."
  description = "Reminder string. Not a binding estimate."
}
