from django.db import models
 
# Create your models here.

class Vehicle_owners(models.Model):
    plate_number = models.CharField(max_length=15)
    owner_id = models.IntegerField()

    class Meta:
        db_table = "vehicle_owners"

    def __str__(self):
        return self.plate_number
 
class Owners(models.Model):
    name = models.CharField(max_length=50)
    mobile = models.CharField(max_length=15)
    mail = models.CharField(max_length=50)
    dept = models.CharField(max_length=25)
    role = models.CharField(max_length=20)

    class Meta:
        db_table = "owners"

    def __str__(self):
        return self.name
