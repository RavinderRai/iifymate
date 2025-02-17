output "db_endpoint" {
  value = aws_db_instance.iifymate_db.endpoint
}

output "db_name" {
  value = aws_db_instance.iifymate_db.db_name
}
