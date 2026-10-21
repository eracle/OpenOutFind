from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('outfind_crm', '0003_pendingqualification'),
    ]

    operations = [
        migrations.AddField(
            model_name='company',
            name='employees_min',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='company',
            name='employees_max',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
    ]
