#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {
	constexpr float camera_fov = 70.0f;
	constexpr float camera_near_plane = 0.01f;
	constexpr float camera_far_plane = 100.0f;

	struct Matrix {
		float m[4][4];
	};

	struct Vector {
		float x, y, z;
	};

	struct Vertex {
		Vector position;
		Vector color;
	};

	struct ShaderConstants {
		Matrix projection;
		Matrix transform;
		Vector color;
	};

	struct VulkanBuffer {
		VkBuffer buffer;
		VkDeviceMemory memory;
	};

	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	VulkanBuffer vertex_buffer{};
	VulkanBuffer index_buffer{};

	std::vector<Vertex> g_vertices;
	std::vector<uint32_t> g_indices;
	uint32_t g_index_count = 0;

	Vector model_position = { 0.0f, 0.0f, 3.0f };
	float model_rotation = 0.0f;
	float model_scale = 1.0f; // model scaling factor
	Vector model_color = { 0.5f, 1.0f, 0.7f };
	bool model_spin = true;

	float camera_yaw = 0.0f; // left-right
	float camera_pitch = 0.0f; // up-down

	Matrix zeroMatrix() {
		Matrix r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				r.m[i][j] = 0.0f;
		return r;
	}

	Matrix identity() {
		Matrix result = zeroMatrix();
		result.m[0][0] = 1.0f;
		result.m[1][1] = 1.0f;
		result.m[2][2] = 1.0f;
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix projection(float fov, float aspect_ratio, float near, float far) {
		Matrix result = zeroMatrix();

		const float radians = fov * M_PI / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);

		result.m[0][0] = cot / aspect_ratio;
		result.m[1][1] = cot;
		result.m[2][3] = 1.0f;

		result.m[2][2] = far / (far - near);
		result.m[3][2] = (-near * far) / (far - near);

		return result;
	}

	Matrix translation(Vector vector) {
		Matrix result = identity();
		result.m[3][0] = vector.x;
		result.m[3][1] = vector.y;
		result.m[3][2] = vector.z;
		return result;
	}

	Matrix scaleMatrix(float s) {
		Matrix result = zeroMatrix();
		result.m[0][0] = s; // scale X
		result.m[1][1] = s; // scale Y
		result.m[2][2] = s; // scale Z
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix rotation(Vector axis, float angle) {
		Matrix result = zeroMatrix();

		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
		if (length == 0.0f) return identity();

		axis.x /= length;
		axis.y /= length;
		axis.z /= length;

		float sina = sinf(angle);
		float cosa = cosf(angle);
		float cosv = 1.0f - cosa;

		result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
		result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

		result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
		result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

		result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

		result.m[3][3] = 1.0f;

		return result;
	}

	Matrix multiply(const Matrix& a, const Matrix& b) {
		Matrix result = zeroMatrix();
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				for (int k = 0; k < 4; k++) {
					result.m[j][i] += a.m[j][k] * b.m[k][i];
				}
			}
		}
		return result;
	}

	// NOTE: Loads shader byte code from file
	VkShaderModule loadShaderModule(const char* path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			return nullptr;
		}
		size_t size = static_cast<size_t>(file.tellg());
		if (size == 0) {
			file.close();
			return nullptr;
		}
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char*>(buffer.data()), size);
		file.close();

		VkShaderModuleCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = buffer.data(),
		};

		VkShaderModule result;
		if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
			return nullptr;
		}

		return result;
	}

	VulkanBuffer createBuffer(size_t size, void* data, VkBufferUsageFlags usage) {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

		VulkanBuffer result{};

		{
			VkBufferCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan buffer\n";
				return {};
			}
		}

		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];
			if ((requirements.memoryTypeBits & (1u << i)) &&
				(type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		VkMemoryAllocateInfo infoAlloc{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &infoAlloc, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan buffer memory\n";
			return {};
		}

		void* device_data = nullptr;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);
		if (device_data && data && size > 0) {
			memcpy(device_data, data, size);
		}
		vkUnmapMemory(device, result.memory);

		return result;
	}

	void destroyBuffer(const VulkanBuffer& buffer) {
		VkDevice& device = veekay::app.vk_device;
		if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
		if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
	}

	void initialize() {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

		{ // NOTE: Build graphics pipeline
			vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
			if (!vertex_shader_module) {
				std::cerr << "Failed to load Vulkan vertex shader from file\n";
				veekay::app.running = false;
				return;
			}

			fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
			if (!fragment_shader_module) {
				std::cerr << "Failed to load Vulkan fragment shader from file\n";
				veekay::app.running = false;
				return;
			}

			VkPipelineShaderStageCreateInfo stage_infos[2];

			stage_infos[0] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertex_shader_module,
				.pName = "main",
			};

			stage_infos[1] = VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragment_shader_module,
				.pName = "main",
			};

			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			VkVertexInputAttributeDescription attributes[] = {
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vertex, position),
				},
				{
					.location = 1,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vertex, color),
				},
			};

			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			VkPipelineMultisampleStateCreateInfo sample_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
				.sampleShadingEnable = false,
				.minSampleShading = 1.0f,
			};

			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = static_cast<float>(veekay::app.window_width),
				.height = static_cast<float>(veekay::app.window_height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			VkRect2D scissor{
				.offset = {0, 0},
				.extent = {veekay::app.window_width, veekay::app.window_height},
			};

			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.pViewports = &viewport,
				.scissorCount = 1,
				.pScissors = &scissor,
			};

			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = true,
				.depthWriteEnable = true,
				.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			};

			VkPipelineColorBlendAttachmentState attachment_info{
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
								  VK_COLOR_COMPONENT_G_BIT |
								  VK_COLOR_COMPONENT_B_BIT |
								  VK_COLOR_COMPONENT_A_BIT,
			};

			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.logicOpEnable = false,
				.logicOp = VK_LOGIC_OP_COPY,
				.attachmentCount = 1,
				.pAttachments = &attachment_info
			};

			VkPushConstantRange push_constants{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
							  VK_SHADER_STAGE_FRAGMENT_BIT,
				.size = sizeof(ShaderConstants),
			};

			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_constants,
			};

			if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline layout\n";
				veekay::app.running = false;
				return;
			}

			VkGraphicsPipelineCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stage_infos,
				.pVertexInputState = &input_state_info,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.layout = pipeline_layout,
				.renderPass = veekay::app.vk_render_pass,
			};

			if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			const int stacks = 100;		// vertical divisions
			const int slices = 100;		// horizontal divisions
			const float radius = 1.0f;		// sphere radius

			g_vertices.clear();
			g_indices.clear();

			for (int i = 0; i <= stacks; ++i) {
				float v = float(i) / float(stacks);
				float phi = v * M_PI;
				float sinPhi = sinf(phi);
				float cosPhi = cosf(phi);

				for (int j = 0; j <= slices; ++j) {
					float u = float(j) / float(slices);
					float theta = u * 2.0f * M_PI;

					// parametric equations
					float x = sinPhi * cosf(theta) * radius;
					float y = cosPhi * radius;
					float z = sinPhi * sinf(theta) * radius;

					Vertex vert;
					vert.position = { x, y, z };

					Vertex tmp = {};
					float rcol = (x + 1.0f) * 0.5f;
					float gcol = (y + 1.0f) * 0.5f;
					float bcol = (z + 1.0f) * 0.5f;
					vert.color = { rcol, gcol, bcol };

					g_vertices.push_back(vert);
				}
			}

			// build indices
			for (int i = 0; i < stacks; ++i) {
				for (int j = 0; j < slices; ++j) {
					uint32_t first = uint32_t(i * (slices + 1) + j);
					uint32_t second = first + uint32_t(slices + 1);

					// triangle 1
					g_indices.push_back(first);
					g_indices.push_back(second);
					g_indices.push_back(first + 1);

					// triangle 2
					g_indices.push_back(second);
					g_indices.push_back(second + 1);
					g_indices.push_back(first + 1);
				}
			}

			g_index_count = static_cast<uint32_t>(g_indices.size());

			// create GPU buffers
			if (!g_vertices.empty()) {
				vertex_buffer = createBuffer(g_vertices.size() * sizeof(Vertex),
					g_vertices.data(),
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			}

			if (!g_indices.empty()) {
				index_buffer = createBuffer(g_indices.size() * sizeof(uint32_t),
					g_indices.data(),
					VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
			}
		}
	}

	void shutdown() {
		VkDevice& device = veekay::app.vk_device;

		// NOTE: Destroy resources here, do not cause leaks in your program!
		destroyBuffer(index_buffer);
		destroyBuffer(vertex_buffer);

		if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
		if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		if (fragment_shader_module) vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		if (vertex_shader_module) vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time) {
		ImGui::Begin("Controls:");
		ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&model_position));
	ImGui::SliderFloat("Rotation", &model_rotation, 0.0f, 2.0f * M_PI);
		ImGui::Checkbox("Spin?", &model_spin);
		ImGui::SliderAngle("Camera Yaw", &camera_yaw, -180.0f, 180.0f);
		ImGui::SliderAngle("Camera Pitch", &camera_pitch, -89.0f, 89.0f);
		ImGui::Text("Scale animation: 1.0 + 0.25 * sin(time)");
		ImGui::End();


		if (model_spin) {
			model_rotation = float(time);
		}

		model_scale = 1.0f + 0.25f * sinf(float(time));

	model_rotation = std::fmod(model_rotation, 2.0f * M_PI);
	if (model_rotation < 0.0f) model_rotation += 2.0f * M_PI;
	}

	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
		vkResetCommandBuffer(cmd, 0);

		{ // NOTE: Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};

			vkBeginCommandBuffer(cmd, &info);
		}

		{ // NOTE: Use current swapchain framebuffer and clear it
			VkClearValue clear_color{ .color = {{0.1f, 0.1f, 0.1f, 1.0f}} };
			VkClearValue clear_depth{ .depthStencil = {1.0f, 0} };

			VkClearValue clear_values[] = { clear_color, clear_depth };

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {
					.extent = {
						veekay::app.window_width,
						veekay::app.window_height
					},
				},
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		// NOTE: Vulkan rendering code here
		{
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
			vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

			Matrix matScale = scaleMatrix(model_scale);
			Matrix matRotate = rotation({ 0.0f, 1.0f, 0.0f }, model_rotation);
			Matrix matTranslate = translation(model_position);

			Matrix modelMat = multiply(matRotate, multiply(matScale, matTranslate));

			Matrix camYaw = rotation({ 0.0f, 1.0f, 0.0f }, -camera_yaw); // left-right
			Matrix camPitch = rotation({ 1.0f, 0.0f, 0.0f }, -camera_pitch); // up-down
			Matrix viewMat = multiply(camPitch, camYaw);

			Matrix transform = multiply(viewMat, modelMat);

			ShaderConstants constants{
				.projection = projection(camera_fov,
										 float(veekay::app.window_width) / float(veekay::app.window_height),
										 camera_near_plane, camera_far_plane),
				.transform = transform,
				.color = model_color,
			};

			vkCmdPushConstants(cmd, pipeline_layout,
				VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(ShaderConstants), &constants);

			if (g_index_count > 0) {
				vkCmdDrawIndexed(cmd, g_index_count, 1, 0, 0, 0);
			}
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
		});
}
