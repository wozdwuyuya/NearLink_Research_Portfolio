# grading/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth import get_user_model
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import reverse, path
from .models import (
    User,
    ClassGroup,
    StudentProfile,
    Exam,
    Question,
    GradingRecord,
    KnowledgePoint,
    OperationLog,
    StudentExam,
    Subject,
    StudentAnswer
)
from django.db.models import Count
from django import forms
from django.core.exceptions import ValidationError
import json
from django_json_widget.widgets import JSONEditorWidget
from django.db import models
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db import transaction
from .views import input_student_answers

User = get_user_model()


# ==== 新增部分：通过学号添加学生答案的表单和Admin ====
class StudentAnswerForm(forms.ModelForm):
    student_id = forms.CharField(label='学生学号', help_text='输入学生的学号')
    exam = forms.ModelChoiceField(queryset=Exam.objects.all(), label='考试')
    question = forms.ModelChoiceField(queryset=Question.objects.all(), label='题目')
    answer = forms.CharField(widget=forms.Textarea, label='学生答案')

    class Meta:
        model = StudentAnswer
        fields = ['student_id', 'exam', 'question', 'answer']

    def clean_student_id(self):
        student_id = self.cleaned_data['student_id']
        try:
            student = User.objects.get(student_id=student_id, role='student')
            return student
        except User.DoesNotExist:
            raise forms.ValidationError('找不到该学号对应的学生')


@admin.register(StudentAnswer)
class StudentAnswerAdmin(admin.ModelAdmin):
    form = StudentAnswerForm
    list_display = ('student', 'exam', 'question', 'answer')
    list_filter = ('exam', 'question')
    search_fields = ('student__student_id', 'student__username')

    def save_model(self, request, obj, form, change):
        # 将表单中的student_id转换为实际的User对象
        obj.student = form.cleaned_data['student_id']
        super().save_model(request, obj, form, change)

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # 如果是修改现有记录，显示学生学号而不是用户对象
        if obj and obj.student:
            form.base_fields['student_id'].initial = obj.student.student_id
        return form


# ==== 新增部分结束 ====

# ==== 用户管理增强 ====
@admin.register(User)
class CustomUserAdmin(BaseUserAdmin):
    filter_horizontal = ('subjects_taught',)
    list_display = ('username', 'role', 'student_id', 'teacher_id', 'is_active')
    list_filter = ('role', 'is_staff', 'is_superuser', 'date_joined')
    search_fields = ('username', 'email', 'student_id')
    ordering = ('-date_joined',)
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('个人信息', {'fields': ('first_name', 'last_name', 'email')}),
        ('角色信息', {
            'fields': ('role', 'student_id', 'teacher_id'),
            'description': '根据角色填写对应ID'
        }),
        ('权限管理', {
            'fields': ('is_active', 'is_staff', 'is_superuser',
                       'groups', 'user_permissions'),
            'classes': ('collapse',)
        }),
        ('时间信息', {
            'fields': ('last_login', 'date_joined'),
            'classes': ('collapse',)
        }),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'password1', 'password2', 'role'),
        }),
    )

    def save_model(self, request, obj, form, change):
        # 根据角色清理ID字段
        if obj.role == 'student':
            obj.teacher_id = None  # 学生不应有教师ID
        elif obj.role == 'teacher':
            obj.student_id = None  # 教师不应有学生ID
        else:  # admin
            obj.student_id = obj.teacher_id = None
        super().save_model(request, obj, form, change)


# ==== 班级管理 ====
@admin.register(ClassGroup)
class ClassGroupAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject', 'teacher', 'student_count')
    list_filter = ('subject',)
    search_fields = ('name', 'teacher__username')
    filter_horizontal = ('students',)

    def student_count(self, obj):
        return obj.students.count()

    student_count.short_description = '学生人数'


# ==== 学生档案 ====
@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'class_group', 'admission_year')
    list_filter = ('class_group', 'admission_year')
    search_fields = ('user__username', 'user__student_id')


# ==== 知识点管理 ====
@admin.register(KnowledgePoint)
class KnowledgePointAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject', 'parent')
    list_filter = ('subject',)
    search_fields = ('name',)
    ordering = ('subject', 'name')


# ==== 考试管理 ====
class QuestionInline(admin.TabularInline):
    model = Question
    extra = 1
    fields = ('number', 'content', 'points', 'knowledge_points')
    formfield_overrides = {
        models.JSONField: {'widget': JSONEditorWidget},
    }


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = ('name', 'subject', 'exam_date', 'total_score', 'is_active')
    list_filter = ('subject', 'exam_date', 'is_active')
    search_fields = ('name',)
    date_hierarchy = 'exam_date'
    inlines = [QuestionInline]
    actions = ['activate_exams', 'deactivate_exams']
    filter_horizontal = ('students',)

    def total_score(self, obj):
        return obj.questions.aggregate(total=Sum('points'))['total'] or 0

    total_score.short_description = '总分'

    def activate_exams(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, f"已激活{queryset.count()}场考试")

    activate_exams.short_description = "激活所选考试"

    def deactivate_exams(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f"已停用{queryset.count()}场考试")

    deactivate_exams.short_description = "停用所选考试"


# ==== 问题管理 ====
@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('exam', 'number', 'content_preview', 'points')
    list_filter = ('exam',)
    search_fields = ('content',)
    formfield_overrides = {
        models.JSONField: {'widget': JSONEditorWidget},
    }

    def content_preview(self, obj):
        content = json.dumps(obj.content, ensure_ascii=False)
        return content[:50] + '...' if len(content) > 50 else content

    content_preview.short_description = '内容预览'


# ==== 评分记录 ====
@admin.register(GradingRecord)
class GradingRecordAdmin(admin.ModelAdmin):
    list_display = ('student', 'question', 'score', 'grader', 'graded_at')
    list_filter = ('question__exam', 'grader')
    search_fields = ('student__username', 'student__student_id')
    date_hierarchy = 'graded_at'
    readonly_fields = ('graded_at',)


# ==== 操作日志 ====
@admin.register(OperationLog)
class OperationLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'action', 'timestamp', 'target_object')
    list_filter = ('action', 'timestamp')
    search_fields = ('user__username', 'target_object')
    date_hierarchy = 'timestamp'
    readonly_fields = ('timestamp',)


# ==== 学生考试关联 ====
@admin.register(StudentExam)
class StudentExamAdmin(admin.ModelAdmin):
    list_display = ('student', 'exam', 'status', 'total_score')
    list_filter = ('exam', 'status')
    search_fields = ('student__username', 'student__student_id')

    def total_score(self, obj):
        return obj.calculate_total_score()

    total_score.short_description = '总分'


# ==== 科目管理 ====
@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'description_preview')
    search_fields = ('name', 'code')

    def description_preview(self, obj):
        return obj.description[:100] + '...' if len(obj.description) > 100 else obj.description

    description_preview.short_description = '描述预览'