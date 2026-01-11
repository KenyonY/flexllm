"""
Folder processor for MLLM client
专门处理文件夹图像数据的处理器类
"""

import os
from pathlib import Path
from typing import List, Callable, Optional, Any, Union, TYPE_CHECKING

# 使用TYPE_CHECKING避免运行时循环引用
if TYPE_CHECKING:
    from .mllm_client import MllmClientBase, MllmClient


class MllmFolderProcessor:
    """
    文件夹处理器类
    专门处理文件夹内图像文件与MLLM客户端的交互
    """
    
    # 支持的图像格式
    SUPPORTED_IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', 
        '.tiff', '.tif', '.svg', '.ico'
    }
    
    def __init__(self, mllm_client: "MllmClient"):
        """
        初始化文件夹处理器
        
        Args:
            mllm_client: MLLM客户端实例
        """
        self.mllm_client = mllm_client
    
    # 属性委托：委托给mllm_client的核心方法
    @property
    def call_llm(self):
        """委托给mllm_client的call_llm方法"""
        return self.mllm_client.call_llm
    
    @property
    def call_llm_with_selection(self):
        """委托给mllm_client的call_llm_with_selection方法"""
        return self.mllm_client.call_llm_with_selection
    
    @property
    def call_llm_sync(self):
        """委托"""
        return self.mllm_client.call_llm_sync
    
    # 数据预处理工具方法（独立于mllm_client）
    def process_image(self, image_path: str) -> str:
        """
        预处理图像路径
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: 处理后的图片路径
        """
        # 转换为绝对路径
        return os.path.abspath(image_path)
    
    def scan_folder_images(
        self,
        folder_path: str,
        recursive: bool = True,
        max_num: Optional[int] = None,
        extensions: Optional[set] = None
    ) -> List[str]:
        """
        扫描文件夹中的图像文件
        
        Args:
            folder_path: 文件夹路径
            recursive: 是否递归扫描子文件夹，默认为True
            max_num: 最大文件数量限制
            extensions: 支持的文件扩展名集合，默认使用SUPPORTED_IMAGE_EXTENSIONS
            
        Returns:
            List[str]: 图像文件路径列表
            
        Raises:
            ValueError: 当输入参数无效时
            FileNotFoundError: 当文件夹不存在时
        """
        # 验证输入参数
        if not folder_path:
            raise ValueError("文件夹路径不能为空")
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"路径不是文件夹: {folder_path}")
        
        # 使用默认扩展名或用户指定的扩展名
        if extensions is None:
            extensions = self.SUPPORTED_IMAGE_EXTENSIONS
        
        # 转换为小写用于比较
        extensions = {ext.lower() for ext in extensions}
        
        image_files = []
        
        # 扫描文件
        if recursive:
            # 递归扫描所有子文件夹
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file_path).suffix.lower() in extensions:
                        image_files.append(file_path)
                        
                        # 检查数量限制
                        if max_num and len(image_files) >= max_num:
                            break
                if max_num and len(image_files) >= max_num:
                    break
        else:
            # 只扫描当前文件夹
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    image_files.append(str(file_path))
                    
                    # 检查数量限制
                    if max_num and len(image_files) >= max_num:
                        break
        
        # 按文件名排序，确保结果的一致性
        image_files.sort()
        
        print(f"扫描完成: 发现 {len(image_files)} 个图像文件")
        if image_files:
            print(f"示例文件: {image_files[0]}")
        
        return image_files
    
    def _build_image_messages_from_files(
        self,
        image_files: List[str],
        system_prompt: str = "",
        text_prompt: str = "请描述这幅图片",
    ) -> List[List[dict]]:
        """
        从图像文件列表构建消息列表
        
        Args:
            image_files: 图像文件路径列表
            system_prompt: 系统提示词
            text_prompt: 文本提示词
            
        Returns:
            messages_list: 消息列表
        """
        messages_list = []
        
        for image_path in image_files:
            messages = []
            
            # 添加系统提示词（如果提供）
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 处理图像路径
            processed_image_path = self.process_image(image_path)
            
            # 添加用户消息（包含文本提示和图像）
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text_prompt}\n文件路径: {processed_image_path}"},
                    {"type": "image_url", "image_url": {"url": f"file://{processed_image_path}"}},
                ],
            })
            
            messages_list.append(messages)
            
        return messages_list
    
    async def call_folder_images(
        self,
        folder_path: str,
        system_prompt: str = "",
        text_prompt: str = "请描述这幅图片",
        recursive: bool = True,
        max_num: Optional[int] = None,
        extensions: Optional[set] = None,
        use_selection: bool = False,
        n_predictions: int = 1,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        return_image_files: bool = False,
        **kwargs,
    ):
        """
        对文件夹中的图像进行批量请求大模型
        
        Args:
            folder_path: 文件夹路径
            system_prompt: 系统提示词，默认为空
            text_prompt: 文本提示词，默认为"请描述这幅图片"
            recursive: 是否递归扫描子文件夹，默认为True
            max_num: 最大处理图像数量限制
            extensions: 支持的文件扩展名集合，默认使用SUPPORTED_IMAGE_EXTENSIONS
            use_selection: 是否使用选择模式
            n_predictions: 每条消息预测次数（仅在use_selection=True时有效）
            selector_fn: 选择函数（仅在use_selection=True时有效）
            return_image_files: 是否在返回结果中包含图像文件列表，默认为False
            **kwargs: 其他传递给MLLM的参数
            
        Returns:
            如果return_image_files=False: response_list
            如果return_image_files=True: (response_list, image_files)
            
        Raises:
            ValueError: 当输入参数无效时
            FileNotFoundError: 当文件夹不存在时
        """
        # 扫描文件夹获取图像文件
        image_files = self.scan_folder_images(
            folder_path=folder_path,
            recursive=recursive,
            max_num=max_num,
            extensions=extensions
        )
        
        if not image_files:
            print("警告: 未找到任何图像文件")
            if return_image_files:
                return [], []
            else:
                return []
        
        # 构建消息列表
        messages_list = self._build_image_messages_from_files(
            image_files, system_prompt, text_prompt
        )
        
        # 调用MLLM
        if use_selection:
            response_list = await self.call_llm_with_selection(
                messages_list,
                n_predictions=n_predictions,
                selector_fn=selector_fn,
                **kwargs
            )
        else:
            response_list = await self.call_llm(messages_list, **kwargs)
        
        # 根据参数决定返回格式
        if return_image_files:
            return response_list, image_files
        else:
            return response_list
    
    async def call_image_files(
        self,
        image_files: List[str],
        system_prompt: str = "",
        text_prompt: str = "请描述这幅图片",
        use_selection: bool = False,
        n_predictions: int = 1,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        **kwargs,
    ):
        """
        对指定的图像文件列表进行批量请求大模型
        
        Args:
            image_files: 图像文件路径列表
            system_prompt: 系统提示词，默认为空
            text_prompt: 文本提示词，默认为"请描述这幅图片"
            use_selection: 是否使用选择模式
            n_predictions: 每条消息预测次数（仅在use_selection=True时有效）
            selector_fn: 选择函数（仅在use_selection=True时有效）
            **kwargs: 其他传递给MLLM的参数
            
        Returns:
            response_list: 响应列表
        """
        if not image_files:
            print("警告: 图像文件列表为空")
            return []
        
        # 验证文件存在性
        valid_files = []
        for file_path in image_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                print(f"警告: 文件不存在，跳过: {file_path}")
        
        if not valid_files:
            print("警告: 没有有效的图像文件")
            return []
        
        # 构建消息列表
        messages_list = self._build_image_messages_from_files(
            valid_files, system_prompt, text_prompt
        )
        
        # 调用MLLM
        if use_selection:
            return await self.call_llm_with_selection(
                messages_list,
                n_predictions=n_predictions,
                selector_fn=selector_fn,
                **kwargs
            )
        else:
            return await self.call_llm(messages_list, **kwargs) 