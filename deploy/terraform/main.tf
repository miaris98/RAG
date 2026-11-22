# P4.5 Immutable Infrastructure as Code — Terraform Skeleton for Production RAG Cluster
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
  }
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Immutable deployment environment name."
}

variable "rag_gateway_image_digest" {
  type        = string
  default     = "ghcr.io/staff-rag/rag-gateway:1.0.0@sha256:0000000000000000000000000000000000000000000000000000000000000001"
  description = "Immutable container image digest (never mutable :latest)."
}

variable "release_track" {
  type        = string
  default     = "blue"
  description = "Active Blue-Green progressive delivery release track."
}

resource "kubernetes_namespace_v1" "rag_prod" {
  metadata {
    name = "rag-${var.environment}"
    labels = {
      "istio-injection" = "enabled"
      "managed-by"      = "terraform"
    }
  }
}

resource "kubernetes_config_map_v1" "rag_gateway_release" {
  metadata {
    name      = "rag-gateway-release-config"
    namespace = kubernetes_namespace_v1.rag_prod.metadata[0].name
  }

  data = {
    RELEASE_TRACK  = var.release_track
    IMAGE_DIGEST   = var.rag_gateway_image_digest
    ZERO_TRUST_TLS = "strict"
  }
}
