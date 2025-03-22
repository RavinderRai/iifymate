variable "db_username" {
  description = "Database administrator username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database administrator password"
  type        = string
  sensitive   = true
}

variable "my_ip" {
  description = "Your IP address for database access"
  type        = string
  sensitive   = true
}
